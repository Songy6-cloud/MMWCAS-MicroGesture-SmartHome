"""A partial two-frame burst must retry and never enter recognition."""
from __future__ import print_function

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dsp_live_gesture'))
from burst_engine import Engine


def wait_for(predicate, timeout=1.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(.005)
    return False


def main():
    engine = Engine(lambda: None, retry_delay=.01, max_retries=3)
    engine.state = 'ready'
    triggered = []
    engine.trigger = triggered.append
    engine.request_capture()
    first_session = triggered[-1]

    # This reproduces the observed early board failure after only two frames.
    engine.offer({'burst': {'session': first_session, 'captured': 2, 'index': 0,
                            'error': 2, 'phase': 4, 'first_us': 1000000,
                            'last_us': 1070000, 'sample_us': 1070000}})
    assert wait_for(lambda: len(triggered) == 2)
    state = engine.snapshot()
    assert state['phase'] == 'arming'
    assert state['collected'] == 0
    assert state['processed'] == 0
    assert state['result'] is None
    assert state['auto_retry_count'] == 1
    assert triggered[-1] != first_session

    # A disconnect during a partial batch waits for the connection and then
    # retries the complete 28-frame burst instead of returning a partial job.
    engine.disconnected('test disconnect')
    state = engine.snapshot()
    assert state['phase'] == 'reconnecting'
    engine.connection_ready()
    assert wait_for(lambda: len(triggered) == 3)
    state = engine.snapshot()
    assert state['phase'] == 'arming'
    assert state['collected'] == 0
    assert state['processed'] == 0
    assert state['auto_retry_count'] == 2

    print(json.dumps({
        'status': 'PASS',
        'partial_frames_rejected': 2,
        'automatic_retry_triggered': True,
        'disconnect_retry_triggered': True,
        'partial_batch_never_recognized': True,
        'retry_limit': state['auto_retry_limit'],
    }, indent=2))


if __name__ == '__main__':
    main()
