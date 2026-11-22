"""MMWCAS V6.6 gesture recognition with selectable model and web UI."""
import argparse
import struct
from collections import deque
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
from urllib.parse import urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault('NANO_EVAL_BATCH', '1')
if sys.platform.startswith('linux') and os.uname().machine == 'aarch64':
    os.environ.setdefault('OPENBLAS_CORETYPE', 'ARMV8')


def verify_project(root, model_name):
    hashes = json.loads((HERE/'baseline_hashes.json').read_text(encoding='utf-8'))
    required = dict(hashes['common'])
    required.update(hashes[model_name])
    for rel, expected in required.items():
        file = root/rel
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError('Project file changed since validation: ' + rel)


def make_loader(root, device, threads, model_name):
    def load():
        verify_project(root, model_name)
        sys.path.insert(0, str(root)); sys.path.insert(0, str(root/'model'))
        import torch
        from predict import GesturePredictor
        torch.set_num_threads(threads)
        if model_name == 'context17':
            from preprocessing_context17 import InterpolationPreprocessor
            preprocess = InterpolationPreprocessor()
            model = GesturePredictor(
                str(root/'checkpoints/yuqing20_standalone_seed42.pt'),
                method='base', device=None if device == 'auto' else device)
            if model.model.num_classes != 16:
                raise ValueError('Expected the validated Context 17 standalone 16-class checkpoint')
            model.runtime_name = 'Context 17 standalone'
        else:
            from preprocessing_efcir import InterpolationPreprocessor
            preprocess = InterpolationPreprocessor()
            model = GesturePredictor(
                str(root/'checkpoints/efcir_seed42.pt'), method='efcir', mode='soft',
                device=None if device == 'auto' else device)
            if model.modules['base'].num_classes != 12 or model.modules['bank'].num_classes != 16:
                raise ValueError('Expected the validated 16-class EF-CIR seed42 checkpoint')
            model.runtime_name = 'EF-CIR soft routing'
        labels = json.loads((root/'labels.json').read_text(encoding='utf-8'))
        return preprocess, model, {str(i):labels[str(i)] for i in range(16)}
    return load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='192.168.33.180')
    p.add_argument('--port', type=int, default=29172)
    p.add_argument('--http-port', type=int, default=8765)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda:0'], default='auto')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--model', choices=['context17', 'efcir'], default='context17',
                   help='recognition model (default: context17)')
    p.add_argument('--ui', choices=['compact', 'full'], default='compact',
                   help='web page: compact Smart Room or full four-plot view')
    p.add_argument('--project', default=str(ROOT))
    a = p.parse_args()
    if a.threads < 1:
        p.error('threads >= 1 required')
    root = Path(a.project).resolve()
    verify_project(root, a.model)
    import numpy as np
    from burst_engine import Engine
    from view_rd import State, Server, Handler, reader
    engine = Engine(make_loader(root, a.device, a.threads, a.model))

    class LiveState(State):
        socket=None
        latest_burst=None
        def attach_socket(self,sock):
            with self.lock:self.socket=sock

        def trigger(self,session):
            with self.lock:
                if not self.connected or self.socket is None or self.latest_burst is None:
                    raise ValueError('需要板端 V6.6 连续采集固件；V6.5 不支持两秒缓存')
                self.socket.sendall(b'BURST28!'+struct.pack('<I',session))

        def accept(self, frame):
            with self.lock:self.latest_burst=frame.get("burst")
            engine.connection_ready()
            engine.offer(frame)
            super().accept(frame)

        def disconnected(self, error):
            with self.lock:self.socket=None;self.latest_burst=None
            super().disconnected(error)
            engine.disconnected(error)

        def snapshot(self, revision=None):
            value = super().snapshot(revision)
            value['recognition'] = engine.snapshot()
            value['runtime_config'] = {'model':a.model, 'ui':a.ui}
            with self.lock:value['burst_capable']=self.latest_burst is not None;value['board_burst']=self.latest_burst
            return value

    class LiveHandler(Handler):
        def do_GET(self):
            path=urlparse(self.path).path
            if path == '/':
                page = 'ui_compact.html' if a.ui == 'compact' else 'ui_full.html'
                self.respond(200, (HERE/page).read_bytes(), 'text/html; charset=utf-8')
            else:
                super().do_GET()

        def do_POST(self):
            path=urlparse(self.path).path
            if path not in ('/api/save-gesture', '/api/start-gesture'):
                super().do_POST(); return
            hosts = ('127.0.0.1:'+str(self.server.server_port), 'localhost:'+str(self.server.server_port))
            origin = self.headers.get('Origin')
            if self.headers.get('Host') not in hosts or (origin and origin not in ['http://'+h for h in hosts]):
                self.respond(403, {'error':'Local origin required'}); return
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 1024 or self.headers.get('Content-Type') != 'application/json':
                    raise ValueError('Expected small JSON request')
                body=json.loads(self.rfile.read(length).decode('utf-8'))
                if not isinstance(body,dict):raise ValueError('Expected a JSON object')
                if path == '/api/start-gesture':
                    current = self.server.state.snapshot()
                    if not current['connected'] or current['age'] is None or current['age'] > 2.:
                        raise ValueError('雷达未连接或没有新帧，请先检查实时点云')
                if path == '/api/start-gesture':
                    engine.request_capture()
                    self.respond(200, engine.snapshot())
                    return
                arrays, report = engine.export()
                dest = root/'dsp-gesture-results'/datetime.now().strftime('%Y%m%d-%H%M%S-%f')
                dest.mkdir(parents=True, exist_ok=False)
                report['result'].pop('_received', None)
                np.savez_compressed(str(dest/'points.npz'), **arrays)
                (dest/'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
                self.respond(200, {'path':str(dest)})
            except (ValueError, OSError) as exc:
                self.respond(400, {'error':str(exc)})

    state = LiveState(root/'dsp-rd-snapshots', 'LIVE / DSP')
    # Live display retains only the current frame, never an idle 28-frame window.
    state.frames = deque(maxlen=1)
    engine.trigger=state.trigger
    # Bind before loading models/connecting, so a second viewer cannot create
    # a competing radar connection when the first viewer owns this port.
    try:
        server = Server(('127.0.0.1', a.http_port), LiveHandler)
    except OSError as exc:
        raise SystemExit('Cannot open viewer port %d. Stop the old RD/gesture viewer with Ctrl+C first. %s' % (a.http_port, exc))
    server.state = state
    stop = threading.Event()
    thread = threading.Thread(target=reader, args=(state, stop, a.host, a.port), daemon=True)
    engine.start(); thread.start()
    print('Open: http://127.0.0.1:%d' % server.server_port, flush=True)
    print('Model: %s | Web UI: %s' % (a.model, a.ui), flush=True)
    print('V6.6: board-confirmed 28-frame capture -> selected preprocessing -> interpolation120 -> selected classifier. Ctrl+C to stop.', flush=True)
    try:
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set(); server.server_close(); thread.join(4.); engine.close()


if __name__ == '__main__':
    main()
