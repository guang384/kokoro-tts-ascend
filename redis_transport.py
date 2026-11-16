"""进程公共件·传输层：redis-server 自举、认领式/普通队列、取消标记。

proc_common 拆分出的队列传输（唯一跨进程通道）。队列方法内的
_make_redis_client 从本模块全局解析——测试兜底请 patch 本模块。
"""
import logging
import os
import pickle
import queue
import struct
import subprocess
import time

import redis

from config import env_int, _qlog

IPC_REDIS_SOCKET = os.getenv("IPC_REDIS_SOCKET", "/tmp/kokoro-redis.sock")
# 外部 redis 模式（IPC_REDIS_URL=tcp://host:port）：不拉起 server、直连
# ——dev/Windows 验证与多副本共享 redis 用；生产留空走 unix socket 自拉
IPC_REDIS_URL = (os.getenv("IPC_REDIS_URL") or "").strip()
TTS_BATCH_BUCKET_CHARS = env_int("TTS_BATCH_BUCKET_CHARS", minimum=8)  # 认领类宽（音素）
# asr 对齐作业的单组音频秒上限：组由此硬封顶（g2p._chunk_cap 语速感知切句
# + funasr_ts._split_groups 超长硬切两层保证），asr 显存预算按它算最坏
# encoder 帧数（pipeline._asr_auto_budget_mb；g2p._chunk_cap 取本值−2s 为
# 切句目标）。生产派发点（compute 的 asr_job 构造）显式传本值；
# protocol.asr_job 的 30.0 默认仅兜缺字段。
# 非可调配参数：改它会同时改变切句粒度、切组粒度与预算峰值，需同步评审。
ASR_MAX_GROUP_SEC = 30.0
_CLAIM_HEADER = 23      # member 头部：2B 长度 + 1B 精度 + 20B FIFO 序号
_CLAIM_SCAN_WINDOW = 64 # 认领扫描窗口（最老 64 条）
_CLAIM_POLL_S = 0.01    # 容量门等待与空认领轮询间隔
_REDIS_POLL_S = 0.01   # 出队轮询间隔（无 BLPOP：消息交接延迟 ≤10ms，量级无感）


_SHARED_REDIS = None   # 进程级共享客户端（cancel SET 等无队列归属的操作用）


def decode_member(m):
    """解出 (chunk, seq)：23B 头（2B 长度 + 1B 精度 + 20B FIFO 序号）+ pickle 载荷。

    解析失败/形状不对返回 (None, None)：损坏 member 由调用方走毒块路径
    丢弃计数。容器内所有进程同镜像同版本，队列无跨版本格式问题。"""
    try:
        chunk = pickle.loads(m[_CLAIM_HEADER:])
        seq = int(m[_CLAIM_HEADER - 20:_CLAIM_HEADER].decode())
    except Exception:  # noqa: BLE001  截断/损坏 member
        return None, None
    if not isinstance(chunk, dict) or "req_id" not in chunk:
        return None, None
    return chunk, seq


def _make_redis_client():
    """redis 客户端工厂（每队列每进程一个实例；连接池线程安全）。
    测试经 monkeypatch 注入 fakeredis 实现单进程语义验证。"""
    # socket 超时必须有界：redis-server 进程活着但 socket 挂起时（CANN/
    # 内核抖动、外部 redis 网络分区），无超时的命令会永久阻塞 worker 主
    # 循环或网关事件循环——表现为"心跳还在、请求全挂"，比快速失败更难查。
    if IPC_REDIS_URL:
        url = IPC_REDIS_URL
        if url.startswith("tcp://"):
            url = "redis://" + url[len("tcp://"):]   # 容错：运维惯写 tcp://
        return redis.Redis.from_url(url, decode_responses=False,
                                    socket_timeout=5.0,
                                    socket_connect_timeout=5.0)
    return redis.Redis(unix_socket_path=IPC_REDIS_SOCKET, decode_responses=False,
                       socket_timeout=5.0, socket_connect_timeout=5.0)


def shared_redis():
    """进程级共享 redis 客户端（懒建；无队列归属的轻量操作用）。"""
    global _SHARED_REDIS
    if _SHARED_REDIS is None:
        _SHARED_REDIS = _make_redis_client()
    return _SHARED_REDIS


def cancel_add(req_id: str) -> None:
    """取消标记（gateway 断连 watcher 与推理超时路径调用，ZSET score=标记时刻）：
    tts worker 认领后、前向前 ZSCORE 检查跳过（g2p 侧不检查——已终止请求
    的 chunk 照常入队，由认领侧统一过滤）。标记寿命 600s（≥ 队列排空
    余量），由 TtsClaimQueue.sweep 周期清扫（有界不泄漏）。"""
    shared_redis().zadd("kokoro:cancel", {req_id: time.time()})


# 各段队列的固定序号镜像（supervisor.create_mailboxes 递增次序：g2p_queue=0
# → tts=1 → asr=2 → gateway_queue_i=3+；Redis 键 kokoro:q:{idx}:{name} 由序号
# 决定，supervisor 与子进程必须一致——改序号两处同步，对应关系由
# test_supervisor_wiring.test_queue_index_mirror_matches_mailboxes 锁定）
QUEUE_INDEX = {"g2p_queue": 0, "tts": 1, "asr": 2}


def claim_queue_pressure(index: int, name: str) -> tuple[int, float]:
    """认领式队列当前压力 (depth=ZCARD, avg_wait_ms=EMA)：网关准入预算
    判断用（网关进程不持有 tts/asr 队列句柄，只能按键直读）。"""
    client = shared_redis()
    base = f"kokoro:q:{index}:{name}"
    pipe = client.pipeline(transaction=False)
    pipe.zcard(base)
    pipe.get(base + ":avg")
    depth, avg = pipe.execute()
    return int(depth or 0), float(avg or 0)


def start_redis_server():
    """拉起 redis-server（unix socket、纯内存、有界）并等就绪。

    supervisor 启动期调用，fail-fast：10s 内 ping 不通直接抛 RuntimeError
    （上层直接退出，绝不出现"服务起来了、队列却是黑洞"的空心态）。
    IPC_REDIS_URL 设置时走外部 redis：不拉起 server（返回 None），仅 ping
    就绪门。"""
    if IPC_REDIS_URL:
        client = _make_redis_client()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                if client.ping():
                    return None
            except Exception:  # noqa: BLE001
                time.sleep(0.05)
        raise RuntimeError(f"external redis 10s 未就绪 url={IPC_REDIS_URL}")
    proc = subprocess.Popen(
        ["redis-server", "--unixsocket", IPC_REDIS_SOCKET,
         "--unixsocketperm", "700", "--port", "0",
         "--save", "", "--appendonly", "no",
         "--maxmemory", "256mb", "--maxmemory-policy", "noeviction"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    client = _make_redis_client()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            if client.ping():
                return proc
        except Exception:  # noqa: BLE001  就绪前 ping 失败属预期
            time.sleep(0.05)
    proc.kill()
    raise RuntimeError(f"redis-server 10s 未就绪 socket={IPC_REDIS_SOCKET}")


class _RedisQueueBase:
    """两队列类的公共底座：键命名、spawn 传递、惰性连接、入队戳、EMA。

    spawn 传递：只传键名元数据，连接在子进程按需重建（子类追加自己的
    额外键，__setstate__ 直接继承）。"""

    _EMA_ALPHA = 0.2

    def __init__(self, maxsize: int, index: int, name: str):
        self._index = int(index)
        self._name = name
        self._cap = int(maxsize) if maxsize else 1024
        self._key = f"kokoro:q:{self._index}:{self._name}"
        self._key_in = self._key + ":in"
        self._key_out = self._key + ":out"
        self._key_avg = self._key + ":avg"   # 近期平均排队 ms（EMA，服务端键，跨消费者共享）
        self._client = None

    # ── spawn 传递：只传键名元数据，连接在子进程按需重建 ──
    def __getstate__(self):
        return {"_index": self._index, "_name": self._name, "_cap": self._cap,
                "_key": self._key, "_key_in": self._key_in,
                "_key_out": self._key_out, "_key_avg": self._key_avg}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._client = None

    def _redis(self):
        if self._client is None:
            self._client = _make_redis_client()
        return self._client

    def _stamp(self, item):
        if isinstance(item, dict):
            item["_q_ts"] = time.time()
        return item

    def _ema_update(self, client, w: float) -> None:
        """近期平均排队 ms（EMA，消费者多写者丢样本仅影响均值精度）。"""
        cur = float(client.get(self._key_avg) or 0)
        client.set(self._key_avg, w if cur <= 0
                   else (1 - self._EMA_ALPHA) * cur + self._EMA_ALPHA * w)

    def close(self):
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001  幂等
                pass


class InstrumentedQueue(_RedisQueueBase):
    """跨进程有界队列（Redis RPUSH + LPOP，唯一传输）。

    对外语义与历版一致：put 超时抛 queue.Full、get 超时抛 queue.Empty、
    有界容量提供背压。depth（qsize）= LLEN = 队列里真实等待处理的消息数
    （空闲必归零）；in/out 由服务端 INCR 计数（单点原子，永不失配）；
    avg_wait_ms 为 EMA（消费者多写者丢样本仅影响均值精度）。

    崩溃语义：消息常驻 redis-server——worker 死亡时队列内消息零丢失
    （其他 worker 接手）；仅"已 LPOP 正在处理"的 1 条随死亡丢失（该请求
    由入口侧 WAIT_TIMEOUT 超时 504 兜底）。redis-server 死亡 = 队列与
    在途**一起丢失**（消息常驻语义的边界：服务端在内存里）——supervisor
    守护自动重启，队列从零起，在途请求走超时兜底。

    容量门 = LLEN 判满 + RPUSH（非原子：超卖幅度 ≤ 并发生产者数、随消费
    自愈；背压方向安全，偏晚不偏早——不引 Lua 保持零脚本依赖）。
    """

    def __init__(self, maxsize: int = 0, index: int = 0, name: str = ""):
        super().__init__(maxsize, index, name or f"q{index}")

    def _count_get(self, item):
        if isinstance(item, dict):
            ts = item.pop("_q_ts", None)
            if ts is not None:
                w = (time.time() - ts) * 1000
                item["_qw_ms"] = w   # 逐条等待随消息带回（EMA 只留均值，精确值供 /tts 日志阶段分解）
                self._ema_update(self._redis(), w)
        return item

    def _try_push(self, data) -> bool:
        """LLEN 判满 + RPUSH（非原子：超卖 ≤ 并发生产者数、随消费自愈）。"""
        client = self._redis()
        if client.llen(self._key) >= self._cap:
            return False
        client.rpush(self._key, data)   # 尾进 + LPOP 头出 = FIFO（LPUSH 会变成 LIFO 栈）
        client.incr(self._key_in)
        return True

    def put(self, item, *args, **kwargs):
        """入队；满时等待至 timeout（None=无限等空间释放），超时抛 queue.Full。"""
        timeout = kwargs.get("timeout", args[0] if args else None)
        data = pickle.dumps(self._stamp(item), protocol=pickle.HIGHEST_PROTOCOL)
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        while True:
            if self._try_push(data):
                return
            if deadline is not None and time.monotonic() >= deadline:
                raise queue.Full()
            time.sleep(0.02)

    def put_nowait(self, item):
        data = pickle.dumps(self._stamp(item), protocol=pickle.HIGHEST_PROTOCOL)
        if not self._try_push(data):
            raise queue.Full()

    def _recv(self, timeout):
        """出队：LPOP + 轮询（交接延迟 ≤ _REDIS_POLL_S，量级无感）。

        - ConnectionError/TimeoutError（redis-server 重启/挂起窗口）自动
          重试：worker 等待守护拉起而非崩溃；有界超时到期抛 queue.Empty。
        - 坏 pickle（跨版本混跑/外部误写）：丢弃该条并计数告警，**绝不
          向上抛**——g2p/结果泵的 get() 在 try 之外，抛一次就是进程/泵死亡。
        - 首次尝试先于 deadline 检查：get_nowait(timeout=0) 必须能取到
          已在队列里的消息（原实现必抛 Empty）。"""
        client = self._redis()
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        first = True
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if not first and remaining is not None and remaining <= 0:
                raise queue.Empty()
            first = False
            try:
                item = client.lpop(self._key)
            except (redis.exceptions.ConnectionError,
                    redis.exceptions.TimeoutError):
                if remaining is not None and remaining <= _REDIS_POLL_S * 2:
                    raise queue.Empty() from None
                time.sleep(0.2)   # redis-server 重启窗口，守护会拉起
                continue
            if item is not None:
                try:
                    client.incr(self._key_out)
                except (redis.exceptions.ConnectionError,
                        redis.exceptions.TimeoutError):
                    pass   # 指标失败不影响消息交付
                try:
                    return pickle.loads(item)
                except Exception as exc:  # noqa: BLE001  坏消息只丢自身
                    _qlog.warning(
                        "queue %s dropped corrupt message (%d bytes): %s",
                        self._name, len(item), str(exc)[:120])
                    continue
            if remaining is not None:
                time.sleep(min(_REDIS_POLL_S, max(0.0, remaining)))
            else:
                time.sleep(_REDIS_POLL_S)

    def get(self, *args, **kwargs):
        timeout = kwargs.get("timeout", args[0] if args else None)
        return self._count_get(self._recv(timeout))

    def get_nowait(self):
        return self._count_get(self._recv(0))

    def qsize(self):
        return int(self._redis().llen(self._key))

    def depth(self):
        """队列真实等待处理数（LLEN）：
        depth==0 表示没有任何批在等 worker（全部空闲或即将空闲）。"""
        return int(self._redis().llen(self._key))

    def stats(self):
        """depth = LLEN = 真实等待处理数（空闲必归零）；in/out = 服务端
        INCR 累计（受理/交付口径，redis 重启诚实清零）。四读管道一次发
        （4→1 RTT）：/health 探针与网关 fast-reject 准入探测的热路径。"""
        pipe = self._redis().pipeline(transaction=False)
        pipe.llen(self._key)
        pipe.get(self._key_in)
        pipe.get(self._key_out)
        pipe.get(self._key_avg)
        depth, in_c, out_c, avg = pipe.execute()
        return {"depth": int(depth or 0), "in": int(in_c or 0),
                "out": int(out_c or 0),
                "avg_wait_ms": round(float(avg or 0), 2)}

    def close(self):
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001  幂等
                pass


class TtsClaimQueue(_RedisQueueBase):
    """TTS chunk 的认领式队列（Redis ZSET：score=FIFO 序号，
    member=2B 音素数 + 1B 精度 + 20B 序号 + pickle 载荷）。

    攒批动作从「定时窗内组装批」变为
    「worker 认领时按队列快照组批」——队列里此刻有什么、能凑多少，worker
    一次认领全拿走（≤max_batch、同类宽度内）。饱和期自动满批，空闲期
    零攒批延迟，零定时器。

    认领即标记：认领 Lua 脚本内同时 ZADD **per-worker 在途账本**
    {key}:infl:{worker_id}（score=认领时刻）——BRPOPLPUSH 级追踪的
    ZSET 等价物：supervisor 按 worker 死亡**精确重派**（requeue_worker），
    慢 worker 永不被 TTL 误重派（曾致重复前向/部分重叠聚合）；TTL 扫描
    只兜"supervisor 漏检"的极端（2×STUCK_KILL）。at-least-once 保留
    （极端下的重复由网关按 idx 幂等去重）。

    键：{key}（ZSET 主队列）、{key}:infl:{worker_id}（per-worker 在途）、
    kokoro:ratt（HASH 重派计数，跨 worker 保留——毒块语义不变）、
    kokoro:seq（INCR FIFO 序号）、{key}:in/:out/:avg/:bad。
    """

    def __init__(self, maxsize: int = 0, index: int = 0, name: str = "tts"):
        super().__init__(maxsize, index, name)
        self._key_seq = "kokoro:seq"
        self._key_ratt = "kokoro:ratt"
        self._key_cancel = "kokoro:cancel"

    # ── spawn 传递：底座键名 + 本类追加键 ──
    def __getstate__(self):
        state = super().__getstate__()
        state.update({"_key_seq": self._key_seq,
                      "_key_ratt": self._key_ratt,
                      "_key_cancel": self._key_cancel})
        return state

    # ── 取消标记（断连取消链：gateway ZADD 时刻 → tts 认领后检查跳过）──
    def cancelled_set(self, req_ids) -> set:
        """批量断连标记查询（单次 RTT）：一批 chunk 常来自同几个请求，
        逐条 ZSCORE 是 B 次 RTT——pipeline 合一。返回已取消的 req_id 集。"""
        req_ids = list(req_ids)
        if not req_ids:
            return set()
        pipe = self._redis().pipeline(transaction=False)
        for rid in req_ids:
            pipe.zscore(self._key_cancel, rid)
        scores = pipe.execute()
        return {rid for rid, s in zip(req_ids, scores) if s is not None}

    def put_nowait(self, chunk: dict):
        """无阻塞入队：容量满立即 queue.Full（降级路径/辅助投递调用）。
        check+push 非原子：超卖 ≤ 并发生产者数（与 _try_push 同款声明）。

        生产事故回归（20261111）：此类曾缺 put_nowait——ship_batch 派发
        asr_job 时 AttributeError 被降级路径吞成"对齐服务不可用"500，
        dryrun 的 FakeQueue 有 put_nowait 测不出。"""
        client = self._redis()
        if client.zcard(self._key) >= self._cap:
            raise queue.Full()
        member, _data, seq = self._encode(chunk)
        client.zadd(self._key, {member: seq})
        client.incr(self._key_in)

    def put(self, chunk: dict):
        """入队一个 chunk（g2p 推送）：容量门（ZCARD 判满轮询等待=背压
        传导上游）→ ZADD score=FIFO 序号 member=长度头+精度+序号+载荷
        → INCR :in。阻塞语义与经典有界队列一致（满则等待空间释放，无超时）。"""
        client = self._redis()
        while client.zcard(self._key) >= self._cap:
            time.sleep(_CLAIM_POLL_S)
        member, _, seq = self._encode(chunk)
        client.zadd(self._key, {member: seq})   # score=FIFO 序号（纯到达序，禁最短优先）
        client.incr(self._key_in)
        return member

    def _encode(self, chunk: dict) -> tuple:
        """member = 2字节长度头 + 1字节精度 + 20位FIFO序号 + pickle 载荷。
        长度头与精度字节供认领脚本读（Lua string.byte，凑批类 =
        (len//width, dtype) 二元组），序号供同分 FIFO。"""
        seq = int(self._redis().incr(self._key_seq))
        payload = pickle.dumps(self._stamp(chunk), protocol=pickle.HIGHEST_PROTOCOL)
        dtype_id = 1 if chunk.get("dtype") == "fp16" else 0
        member = (struct.pack(">H", len(chunk.get("ps") or ""))
                  + bytes([dtype_id])
                  + f"{seq:020d}".encode() + payload)
        return member, payload, seq

    _CLAIM_SCRIPT = """
        local members = redis.call('ZRANGE', KEYS[1], 0, tonumber(ARGV[4]) - 1)
        if #members == 0 then return {} end
        local claimed = {}
        local width = tonumber(ARGV[3])
        local function hd(m)
            -- 短 member（外部写入/损坏）不能让整个认领脚本抛错——一旦抛错
            -- 每次 claim 都失败，tts 池永久停摆。nil 兜 0 后由 Python 侧丢弃
            return (string.byte(m, 1) or 0) * 256 + (string.byte(m, 2) or 0)
        end
        -- 凑批类 = (音素长度类, 精度)：fp16/fp32 不混批（autocast 整批上下文）
        local ref = hd(members[1])
        local refclass = math.floor(ref / width)
        local refdt = string.byte(members[1], 3) or -1   -- <3 字节：无精度字节
        local now = ARGV[2]
        for i, m in ipairs(members) do
            if #claimed >= tonumber(ARGV[1]) then break end
            local L = hd(m)
            if math.floor(L / width) == refclass
               and (string.byte(m, 3) or -1) == refdt then
                redis.call('ZREM', KEYS[1], m)
                redis.call('ZADD', KEYS[2], now, m)
                claimed[#claimed + 1] = m
            end
        end
        return claimed
    """

    def claim(self, max_batch: int, worker_id: str = ""):
        """认领一批：与 head 同类（len//width 相同）的最多 max_batch 条，
        Lua 原子完成 弹出+在途标记（认领即标记，BRPOPLPUSH 级追踪）。
        worker_id 形如 "tts-0-12345"（role-i-pid）——死亡重派按它精确
        定位在途账本。返回 [(member, chunk_dict), ...]；队列空返回 []。"""
        client = self._redis()
        now = time.time()   # 本批认领时刻（同时用作在途 score 与 q_* 跨度终点）
        members = client.eval(
            self._CLAIM_SCRIPT, 2, self._key,
            f"{self._key}:infl:{worker_id}",
            max_batch, now, TTS_BATCH_BUCKET_CHARS,
            _CLAIM_SCAN_WINDOW)
        out, waits_ms, bad = [], [], []
        for m in members:
            chunk, _seq = decode_member(m)
            if chunk is None:
                # 真损坏（头/载荷解不出）：丢弃并计数——否则该消息每轮被
                # 认领 → 崩 worker → 重派循环（put_nowait 事故同构）。
                bad.append(m)
                continue
            ts = chunk.pop("_q_ts", None)
            if isinstance(ts, (int, float)):   # 损坏的非数值 _q_ts 不能炸 claim
                w = (now - ts) * 1000
                waits_ms.append(w)
                # 请求级排队跨度用：入队时刻 + 本批认领时刻。网关对同请求
                # 全部 chunk 取 min(入队)/max(认领)，得到"第一个进队到最后一个
                # 出队"的墙钟跨度。
                chunk["_q_enq"] = ts
                chunk["_q_claim"] = now
            out.append((m, chunk))
        # 以下全部是"认领已经成功"之后的簿记：任何一步失败都不能让本函数
        # 抛错——member 已在 infl 账本里，调用方只看到异常就会重试 claim，
        # 这批 member 却无人持有（要等 600s TTL 才重派，请求 120s 先 504）。
        # 因此逐段 best-effort：失败只告警，out 始终返回。
        if bad:
            # 一批坏 member 聚合成一条 pipeline（4 命令 1 RTT；旧实现每条
            # 4 次 RTT）。:out 与 :in 同单位——自家 put 的坏块保持收支平衡；
            # 外部直写、从未 :in 的 member 会让 out>in（仅指标观感）。
            try:
                pipe = client.pipeline(transaction=False)
                pipe.zrem(f"{self._key}:infl:{worker_id}", *bad)
                pipe.hdel(self._key_ratt, *bad)
                pipe.incr(self._key_out, len(bad))
                pipe.incr(self._key + ":bad", len(bad))
                pipe.execute()
            except Exception as exc:  # noqa: BLE001  簿记失败不吞已认领 member
                _qlog.warning("claim bad-member cleanup failed worker=%s n=%d: %s",
                              worker_id, len(bad), str(exc)[:120])
        if waits_ms:
            # EMA 一次 GET/SET（旧实现逐 chunk 两 RTT × B）：批均值入 EMA，
            # 长期均值语义不变
            try:
                self._ema_update(client, sum(waits_ms) / len(waits_ms))
            except Exception as exc:  # noqa: BLE001  纯指标，绝不影响交付
                _qlog.warning("claim wait-EMA update failed worker=%s: %s",
                              worker_id, str(exc)[:120])
        if out:
            try:
                client.incr(self._key_out, len(out))  # 与 put 的 in 同单位（chunk 数）
            except Exception as exc:  # noqa: BLE001  纯指标
                _qlog.warning("claim out counter failed worker=%s: %s",
                              worker_id, str(exc)[:120])
        return out

    # 交付确认：逐 member 原子「移除成功才清重派计数」——慢 worker 晚 ack
    # 时不会误清另一个在途副本的 :ratt（竞态需活 worker 被判死；当前
    # supervisor 的 SWEEP_TTL=2×STUCK_KILL 已使该窗口不存在，本脚本把
    # 它从"依赖时间常数"变成"结构上不可能"）
    _ACK_SCRIPT = """
        local n = 0
        for i = 1, #ARGV do
            if redis.call('ZREM', KEYS[1], ARGV[i]) == 1 then
                redis.call('HDEL', KEYS[2], ARGV[i])
                n = n + 1
            end
        end
        return n
    """

    def ack(self, members, worker_id: str = "") -> int:
        """处理完成后移除在途标记（交付确认）。redis 瞬断时标记残留由
        requeue_worker（死亡时）或 TTL 兜底收口。ZREM+HDEL 一次 Lua
        单次 RTT（旧实现逐 member 一次 RTT，B=8 时 8 次往返）。

        同批清掉重派计数（:ratt）：成功交付后计数生命周期结束——旧实现
        只 zrem，每次 worker 死亡重派都会在 hash 里留一条永久 member 记录
        （每死亡 ~在途 chunk 数，容器生命周期内只增不减）。"""
        if not members:
            return 0
        client = self._redis()
        infl_key = f"{self._key}:infl:{worker_id}"
        # Lua 幂等：重试安全（第一次已执行但响应丢失时，第二次 ZREM=0、
        # 不重复清 :ratt）。失败必须留痕——静默返回 0 会让宿主的在途
        # 账本只增不减，直到 TTL 重派（重复前向）才暴露。
        for attempt in (1, 2):
            try:
                return int(client.eval(self._ACK_SCRIPT, 2, infl_key,
                                       self._key_ratt, *members))
            except Exception as exc:  # noqa: BLE001
                if attempt == 1:
                    _qlog.warning("ack failed worker=%s n=%d (retry once): %s",
                                  worker_id, len(members), str(exc)[:120])
                    time.sleep(0.05)
                else:
                    _qlog.warning("ack failed worker=%s n=%d (left to TTL sweep): %s",
                                  worker_id, len(members), str(exc)[:120])
        return 0

    def _drop_member(self, client, infl_key: str, m) -> None:
        """毒块/超限 member：清在途账本+重派计数，:bad 计数（统计与防护
        共用一条路径——两种放弃在 /metrics 观感一致）。"""
        client.zrem(infl_key, m)
        client.hdel(self._key_ratt, m)
        client.incr(self._key + ":bad")

    def _redispatch_one(self, client, infl_key: str, m, seq,
                        max_redispatch) -> str:
        """单 member 重派（requeue_worker/sweep 共用）：毒块（seq=None，
        decode 失败）或重派超限 → 清账本丢弃返回 "gaveup"；正常 → 回主
        队列（score=原 FIFO 序号，保到达序）并移出在途，返回 "ok"。"""
        if seq is None:
            self._drop_member(client, infl_key, m)
            return "gaveup"
        if max_redispatch is not None:
            attempts = client.hincrby(self._key_ratt, m, 1)
            if attempts > max_redispatch:
                self._drop_member(client, infl_key, m)
                return "gaveup"
        client.zadd(self._key, {m: seq})
        client.zrem(infl_key, m)
        return "ok"

    def requeue_worker(self, worker_id: str, max_redispatch: int | None = None) -> int:
        """worker 死亡精确重派：其在途账本的全部 member 立即回主队列
        （score=member 内原 FIFO 序号，保到达序）。supervisor 在 respawn
        处调用——恢复时效 ~秒级（替代"等 TTL 猜"的分钟级慢路径）。

        max_redispatch=None：不计数（无上限；手工/测试默认）。
        max_redispatch>=0：与 sweep 同款毒块防护——hincrby 重派计数，
        超限丢弃；**0 表示不重派**（与 sweep 的首次即弃一致）。
        **worker 死亡主路径必须计数**——supervisor 在 respawn 时调用本
        函数并清空在途账本，sweep 再也看不到这些 member，若只靠 sweep
        计数则"死亡→重派→再死"可无限循环（REDISPATCH_MAX 形同虚设）。"""
        client = self._redis()
        infl_key = f"{self._key}:infl:{worker_id}"
        n = 0
        try:
            for m in client.zrange(infl_key, 0, -1):
                _chunk, seq = decode_member(m)
                if (self._redispatch_one(client, infl_key, m, seq,
                                         max_redispatch) == "ok"):
                    n += 1   # 毒块单条丢弃，不中断整本账本的重派
        except Exception:  # noqa: BLE001  redis 瞬断：下轮 sweep 兜底
            pass
        return n

    def depth(self):
        return int(self._redis().zcard(self._key))

    def stats(self):
        """五读管道一次发（5→1 RTT，与 InstrumentedQueue.stats 同款）。"""
        pipe = self._redis().pipeline(transaction=False)
        pipe.zcard(self._key)
        pipe.get(self._key_in)
        pipe.get(self._key_out)
        pipe.get(self._key + ":bad")
        pipe.get(self._key_avg)
        depth, in_c, out_c, bad, avg = pipe.execute()
        return {"depth": int(depth or 0), "in": int(in_c or 0),
                "out": int(out_c or 0), "bad": int(bad or 0),
                "avg_wait_ms": round(float(avg or 0), 2)}

    def sweep(self, alive_pids: set, now: float, ttl: float,
              max_redispatch: int) -> dict:
        """在途重派扫描（supervisor monitor 每 2s 调用）。

        BRPOPLPUSH 级追踪语义：重派条件是「worker 已死」而非「TTL 超时」
        ——alive_pids 是 supervisor 的存活子进程集合，per-worker 在途账本
        的 pid 不在其中即精确重派（慢 worker 永不被误重派，重复前向/
        部分重叠聚合的触发面从"每次慢批"缩到"极端故障"）；TTL 只兜
        supervisor 漏检的极端（建议 2×STUCK_KILL_S）。重派次数超上限的
        放弃（毒块防护，请求走 WAIT_TIMEOUT 504）。顺带清扫过期断连
        取消标记。

        不变量：**ttl 必须大于活 worker 的最长持握**（supervisor 取
        SWEEP_TTL=2×STUCK_KILL_S，而看门狗 STUCK_KILL_S 会先杀掉超时
        worker）——这是"慢 worker 不被 TTL 误重派"与 ack 精确清计数
        的共同前提；调小 ttl 前先核对这两个常数。"""
        redispatched = gaveup = 0
        client = self._redis()
        client.zremrangebyscore(self._key_cancel, "-inf", now - 600)
        infl_keys = [k for k in client.scan_iter(
            match=f"{self._key}:infl:*", count=256)]
        for infl_key in infl_keys:
            if isinstance(infl_key, bytes):
                infl_key = infl_key.decode()   # redis-py 默认 bytes key
            wid = infl_key.rsplit(":infl:", 1)[-1]
            try:
                pid = int(wid.rsplit("-", 1)[-1])
            except ValueError:
                continue
            # 死者立即全量重派（无条件）；活者的条目仅在超长 TTL 兜底时
            # 重派（inflight score=认领时刻，正常处理 << ttl）
            if pid in alive_pids:
                members = client.zrangebyscore(infl_key, "-inf", now - ttl,
                                               start=0, num=100)
            else:
                members = client.zrange(infl_key, 0, -1)
            for m in members:
                # seq = member 内原 FIFO 序号（F4：曾用 ps_len 作 score 致插队）。
                # 毒块单条丢弃不抛出——否则整轮 sweep 中止，同轮其他死亡
                # worker 的在途 chunk 全部延迟到下一轮（甚至永不重派）。
                _chunk, seq = decode_member(m)
                if (self._redispatch_one(client, infl_key, m, seq,
                                         max_redispatch) == "ok"):
                    redispatched += 1
                else:
                    gaveup += 1
        return {"redispatched": redispatched, "gaveup": gaveup}


    def close(self):
        client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001  幂等
                pass


def mailbox_depths(queues: dict) -> dict:
    """Mailbox 统计探针：{名字: {depth, in, out, avg_wait_ms}}，列表值
    展开为 name[i]。depth = LLEN/ZCARD = 队列真实长度（等待处理数，
    空闲归零）。

    任何异常（队列已关闭/redis 瞬断等）一律 None——心跳线程靠它活着，
    这里抛异常会让 worker 心跳停写而被 supervisor 误杀。历史事故：
    gateway_queues 是列表，直接 .stats() 会 AttributeError 且未被捕获
    → supervisor 心跳从未落盘，/health 的缺员诊断因此长期失效。
    """
    out = {}
    for name, q in queues.items():
        if isinstance(q, (list, tuple)):
            for i, sub in enumerate(q):
                out[f"{name}[{i}]"] = _queue_stats(sub)
        else:
            out[name] = _queue_stats(q)
    return out



def _queue_stats(q) -> dict:
    """单队列统计：InstrumentedQueue/TtsClaimQueue 给全量；普通队列只有 depth。"""
    try:
        if isinstance(q, (InstrumentedQueue, TtsClaimQueue)):
            return q.stats()
        return {"depth": q.qsize()}
    except Exception:  # noqa: BLE001  平台不支持/队列已关闭 → unknown
        return {"depth": None}
