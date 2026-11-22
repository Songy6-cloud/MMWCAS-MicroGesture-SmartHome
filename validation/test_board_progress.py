"""Deterministic check that web progress is sourced only from the active board session."""
from __future__ import print_function

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dsp_live_gesture'))
from burst_engine import Engine


class Clock:
    value = 100.0

    def __call__(self):
        return self.value


def main():
    clock = Clock()
    engine = Engine(lambda: None, clock=clock)
    engine.state = 'ready'
    triggered = []
    engine.trigger = triggered.append
    engine.request_capture()
    session = triggered[0]
    first_us, last_us = 1000000, 2890000

    # A packet from an older session must not advance the displayed count.
    engine.offer({'burst': {'session': session - 1, 'captured': 27, 'index': 0,
                            'error': 0, 'phase': 1, 'first_us': first_us,
                            'last_us': last_us, 'sample_us': first_us}})
    assert engine.snapshot()['collected'] == 0

    observed = []
    for captured in range(1, 29):
        engine.offer({'burst': {'session': session, 'captured': captured, 'index': 0,
                                'error': 0, 'phase': 1 if captured < 28 else 2,
                                'first_us': first_us, 'last_us': last_us,
                                'sample_us': first_us}})
        snapshot = engine.snapshot()
        observed.append(snapshot['collected'])
        assert snapshot['collected'] == captured
        assert snapshot['progress_source'] == 'board_burst_header'
        assert snapshot['progress_session_verified'] is True

    for ordinal in range(1, 29):
        sample_us = first_us + (ordinal - 1) * 70000
        frame = {
            'frame_id': 900 + ordinal,
            'channel': 0,
            'dsp_id': 0,
            'xyz': np.zeros((1, 3), dtype=np.float64),
            'rd': {'version': 2, 'selection_enabled': 2},
            'burst': {'session': session, 'captured': 28, 'index': ordinal,
                      'error': 0, 'phase': 3, 'first_us': first_us,
                      'last_us': last_us, 'sample_us': sample_us},
        }
        engine.offer(frame)
        assert engine.snapshot()['processed'] == ordinal

    report = {
        'status': 'PASS',
        'progress_source': 'board_burst_header',
        'wrong_session_did_not_advance': True,
        'observed_capture_counts': observed,
        'processed_counts_exact': True,
        'capture_span_seconds': engine.snapshot()['capture_span_seconds'],
    }
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
