import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from ta2.worker import Publisher


class WorkerTests(unittest.TestCase):
    def test_disabled_publisher_never_connects(self):
        with tempfile.TemporaryDirectory() as folder, patch('ta2.publication.connect_router') as connect:
            worker=Publisher(Path(folder),'s1',object(),False).start()
            worker.notify(); worker.close()
            connect.assert_not_called()
    def test_network_io_is_independent_of_training_thread(self):
        entered=threading.Event(); release=threading.Event()
        class Outbox:
            def publish(self,**kwargs):
                entered.set(); release.wait(2)
                return dict(pending_records=0,errors=[],status='UPLOAD_VERIFIED')
        with tempfile.TemporaryDirectory() as folder, patch('ta2.publication.connect_router',return_value=object()):
            worker=Publisher(Path(folder),'s1',Outbox(),True).start()
            self.assertTrue(entered.wait(2))
            self.assertTrue(worker.thread.is_alive())
            release.set(); worker.close()
            self.assertFalse(worker.thread.is_alive())


if __name__=='__main__':unittest.main()
