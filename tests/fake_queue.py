"""线程安全的有界队列替身：单进程内驱动 actor 主循环（g2p/asr 等）。"""
import queue as _pyq
import threading
import time


class FakeQueue:
    """put_nowait/put(timeout)/get_nowait/get(timeout)/qsize/close 语义对齐 mp.Queue。"""

    def __init__(self, maxsize: int = 0):
        self._items = []
        self._max = maxsize
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self.closed = False

    def put_nowait(self, item):
        with self._not_full:
            if self.closed:
                raise ValueError("Queue is closed")
            if self._max and len(self._items) >= self._max:
                raise _pyq.Full()
            self._items.append(item)
            self._not_empty.notify()

    def put(self, item, timeout=None, block=True):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_full:
            while True:
                if self.closed:
                    raise ValueError("Queue is closed")
                if not self._max or len(self._items) < self._max:
                    self._items.append(item)
                    self._not_empty.notify()
                    return
                if deadline is None:
                    self._not_full.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _pyq.Full()
                    self._not_full.wait(remaining)

    def get_nowait(self):
        with self._not_empty:
            if not self._items:
                raise _pyq.Empty()
            item = self._items.pop(0)
            self._not_full.notify()  # 唤醒阻塞在满队 put 上的生产者（对齐 mp.Queue 语义）
            return item

    def get(self, timeout=None, block=True):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._not_empty:
            while True:
                if self._items:
                    item = self._items.pop(0)
                    self._not_full.notify()  # 同上
                    return item
                if deadline is None:
                    self._not_empty.wait()
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise _pyq.Empty()
                    self._not_empty.wait(remaining)

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)

    def depth(self) -> int:
        """对齐 InstrumentedQueue.depth() 语义（真实等待处理数）。"""
        return self.qsize()

    def stats(self) -> dict:
        """对齐 InstrumentedQueue.stats() 键形（网关 fast_reject 准入闸读取；
        替身恒无排队压力——需要压力的用例直接预置 redis 键）。"""
        return {"depth": self.qsize(), "in": 0, "out": 0, "avg_wait_ms": 0.0}

    def close(self):
        with self._lock:
            self.closed = True


class Runner:
    """守护线程里跑 actor 主循环；捕获异常供断言（线程崩溃 = 测试失败）。"""

    def __init__(self, target, *args):
        self.error = None
        self._t = threading.Thread(target=self._wrap, args=(target, *args),
                                   daemon=True)

    def _wrap(self, target, *args):
        try:
            target(*args)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.error = "".join(traceback.format_exception(exc))

    def start(self):
        self._t.start()
        return self

    def alive(self) -> bool:
        return self._t.is_alive()

    def fail_reason(self):
        return self.error
