"""Optional bounded I/O publisher; network latency never stalls an optimizer."""
import threading

from .common import atomic_json, now


class Publisher:
    def __init__(self,lane,server,outbox,enabled=False):
        self.lane,self.server,self.outbox=lane,server,outbox
        self.enabled=enabled; self.stop=threading.Event(); self.wake=threading.Event()
        self.thread=None

    def start(self):
        if self.enabled:
            self.thread=threading.Thread(target=self._run,name='ta2-append-only-publisher',daemon=True)
            self.thread.start()
        return self

    def notify(self):
        self.wake.set()

    def _run(self):
        router=None
        while not self.stop.is_set():
            try:
                if router is None:
                    from .publication import connect_router
                    router=connect_router(self.lane,self.server)
                state=self.outbox.publish(adapter=router,max_records=100)
                atomic_json(self.lane/'upload_status.json',state)
                # HTTPClient enforces persisted per-request quota independent
                # of notify(). An entirely deferred queue needs no busy poll.
                wait=30 if state.get('errors') or (state.get('pending_records') and
                    not state.get('attempted_this_batch', 1)) else 1 if state.get('pending_records') else 10
            except Exception as exc:
                atomic_json(self.lane/'upload_status.json',dict(status='PARTIAL_OR_PENDING_PUBLICATION',
                    error=str(exc),checked_utc=now(),experiment_complete=False))
                wait=30
            self.wake.wait(wait); self.wake.clear()

    def close(self):
        self.stop.set(); self.wake.set()
        if self.thread:
            self.thread.join(timeout=2)
        # Any interrupted append is resolved by Record_ID readback on next run;
        # no receipt is forged and payloads stay in the durable outbox.
