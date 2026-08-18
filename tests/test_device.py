import pytest
import simpy
from opal.infra.io_model import AbstractDevice, OpalIORequest
from opal.core.environment import OpalSimulatorEnvironment

NANO_SECONDS = 1e9
GIB = 2**30
MIB= 2**20
BW = 100 * GIB # unit in bytes per second
LATENCY_SEC = 20 / 1000 # 20 ms in seconds
CAPACITY = 1000 * GIB
CONCURRENCY = 1000
REQUEST_COUNT = 1000
REQUEST_SIZE = 80 * MIB
#REQUEST_SIZE = BW * LATENCY_SEC # NOTE: this has the largest effect

@pytest.fixture
def fake_env():
    class MockEnv:
        def __init__(self):
            self.simpy_env = simpy.Environment()
        def get_config(self): return {}
        def are_we_done(self): return False
    return MockEnv()

class TestAbstractDevice:
    def test_measured_bw(self, fake_env:OpalSimulatorEnvironment):
        """
        Test that measured bandwidth meets configured bandwidth
        """
        dev = AbstractDevice(fake_env, "fake_device", CAPACITY, BW, LATENCY_SEC, CONCURRENCY)
        requests = [OpalIORequest(REQUEST_SIZE) for _ in range(REQUEST_COUNT)]
        dev.process_requests(requests)
        env = fake_env.simpy_env
        env.run(1000) # arbitrary
        for r in range(10):
            print(requests[r].finish_time)
        # compute max finish time to get duration
        duration = max(r.finish_time for r in requests)
        measured_bw = REQUEST_COUNT * REQUEST_SIZE / duration
        print(f"measured bw and config bw should have a ratio <= 1: {measured_bw / BW}")
        assert measured_bw / BW <= 1