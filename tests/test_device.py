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
    def test_latency_dominated(self, fake_env:OpalSimulatorEnvironment):
        """
        Test a small I/O such that its duration (processing time) is dominated by its latency and not BW
        """
        dev = AbstractDevice(fake_env, "fake_device", CAPACITY, BW, LATENCY_SEC, CONCURRENCY)
        request = OpalIORequest(1 * 1024) # 1KiB
        dev.process_one_request(request)
        env = fake_env.simpy_env
        env.run(1000) # arbitrary
        duration = request.finish_time - request.arrival_time
        # 1KiB transfers in ~9.5ns against 20ms of latency, so latency should
        # account for essentially the whole duration
        assert duration >= LATENCY_SEC
        assert duration < LATENCY_SEC * 1.01

    def test_bandwidth_bound(self, fake_env:OpalSimulatorEnvironment):
        """
        Test large requests take longer than latency
        """
        dev = AbstractDevice(fake_env, "fake_device", CAPACITY, BW, LATENCY_SEC, CONCURRENCY)
        request_size = 100 * GIB
        request = OpalIORequest(request_size)
        dev.process_one_request(request)
        env = fake_env.simpy_env
        env.run(1000) # arbitrary
        duration = request.finish_time - request.arrival_time
        # latency is additive on top of the transfer, not overlapped with it
        assert duration == pytest.approx(LATENCY_SEC + request_size / BW, rel=1e-6)

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

    def test_interrupt_bw(self, fake_env:OpalSimulatorEnvironment):
        """
        Test the interruption path of the bandwidth manager, and make sure
        that measured bandwdith meets configured bandwidth
        """

        dev = AbstractDevice(fake_env, "fake_device", CAPACITY, BW, LATENCY_SEC, CONCURRENCY)

        # For the problem to appear, we need < CONCURRENCY requests to be in the system first
        # allow that request to run for a few ms, and then insert the next request
        # the next request will be given fake credit, allowing us to measure a faster than configured bw. 

        # the resident request has to still be transferring when the second one
        # registers, so size it to outlast interrupt_delay
        interrupt_delay = 5 / 1000 # 5 ms in seconds
        resident_size = 4 * GIB

        env = fake_env.simpy_env
        resident_request = OpalIORequest(resident_size)
        interrupt_request = OpalIORequest(REQUEST_SIZE)

        def submit_after(request, delay):
            """Hand a request to the device `delay` seconds into the run"""
            yield env.timeout(delay)
            dev.process_one_request(request)

        dev.process_one_request(resident_request)
        env.process(submit_after(interrupt_request, interrupt_delay))

        env.run(1000) # arbitrary

        # service time is latency plus transfer; the transfer alone can never
        # beat size / BW no matter what else is in flight
        transfer_time = interrupt_request.finish_time - interrupt_request.arrival_time - LATENCY_SEC
        fastest_possible = REQUEST_SIZE / BW
        print(f"transfer took {transfer_time*1000:.4f} ms, floor is {fastest_possible*1000:.4f} ms")
        assert transfer_time >= fastest_possible

