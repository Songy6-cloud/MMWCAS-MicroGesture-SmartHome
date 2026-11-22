"""One bounded 28-frame stream feeding the existing Nano preprocessing/model."""
from collections import deque
import copy
import threading
import time
import numpy as np


class Engine:
    def __init__(self, loader, stride=7, clock=time.monotonic):
        if not 1 <= stride <= 28:
            raise ValueError('stride must be 1..28')
        self.loader, self.stride, self.clock = loader, stride, clock
        self.condition = threading.Condition()
        self.frames = deque(maxlen=28)
        self.previous = self.settings = self.last_time = None
        self.since = self.generation = self.skipped = self.id_gaps = 0
        self.pending = self.result = self.saved = None
        self.stop = False
        self.phase = 'idle'
        self.capture_started = None
        self.state, self.message = 'loading', '正在加载手势模型'
        self.busy = False
        self.thread = None

    def _reset(self, message):
        # Caller holds condition. In-flight work is invalidated by generation.
        self.generation += 1
        self.phase = 'idle'
        self.capture_started = None
        self.frames.clear()
        self.previous = self.settings = self.last_time = None
        self.since = 0
        self.id_gaps = 0
        self.pending = self.result = self.saved = None
        self.message = message

    def request_capture(self):
        with self.condition:
            if self.stop or self.state != 'ready':
                raise ValueError('模型尚未就绪，请稍后再试')
            if self.phase in ('collecting', 'processing') or self.busy:
                raise ValueError('本次识别尚未结束，请等待结果')
            self._reset('正在收集本次手势，请做动作')
            self.phase = 'collecting'
            self.capture_started = self.clock()

    def disconnected(self, error):
        with self.condition:
            if self.phase in ('collecting', 'processing'):
                self._reset('连接中断，本次识别已取消；恢复连接后请重新点击开始')

    def offer(self, frame, now=None):
        now = self.clock() if now is None else now
        rd = frame.get('rd')
        with self.condition:
            if self.stop or self.state != 'ready' or self.phase != 'collecting':
                return
            if now < self.capture_started:
                return
            if not rd or rd.get('version') != 2 or rd.get('selection_enabled') != 2:
                self._reset('固件不匹配，本次识别已取消；需要 MATLAB 全链 V6 点云')
                return
            key = (frame['channel'], frame['dsp_id'])
            settings = tuple(rd.get(k) for k in
                             ('version', 'min_bin', 'max_bin', 'min_area', 'k0', 'kd', 'selection_enabled'))
            delta = (frame['frame_id'] - self.previous[0]) % 2**32 if self.previous else 1
            if self.previous and key == self.previous[1] and delta == 0:
                return  # Duplicate packet is not a new observation.
            if self.previous is not None and (
                delta >= 2**31 or key != self.previous[1] or
                now - self.last_time > 2. or settings != self.settings
            ):
                self._reset('连接、帧序或设置变化，本次识别已取消，请重新点击开始')
                return
            self.id_gaps += max(0, delta - 1)
            self.previous = (frame['frame_id'], key)
            self.settings, self.last_time = settings, now
            # Exclude RD heatmaps from inference jobs; arrays are owned copies.
            xyz = np.asarray(frame['xyz'], dtype=np.float64).reshape(-1, 3).copy()
            if not np.isfinite(xyz).all():
                self._reset('收到无效坐标，本次识别已取消，请重新点击开始')
                return
            self.frames.append((frame['frame_id'], xyz, now))
            self.message = '正在收集本次手势：%d/28 帧' % len(self.frames)
            if len(self.frames) == 28:
                self.phase = 'processing'
                self.message = '已收满 28 帧，正在滤波、插值和识别'
                self.pending = (self.generation, now, list(self.frames))
                self.condition.notify()

    def start(self):
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def close(self):
        with self.condition:
            self.stop = True
            self._reset('已停止')
            self.condition.notify_all()
        if self.thread:
            self.thread.join(5.)

    def snapshot(self):
        with self.condition:
            now = self.clock()
            if self.phase == 'collecting' and now - (self.last_time if self.last_time is not None else self.capture_started) > 2.:
                self._reset('接收超时，本次识别已取消，请重新点击开始')
            result = dict(self.result) if self.result else None
            if result:
                result['age_seconds'] = max(0., now-result.pop('_received'))
                result['stale'] = result['age_seconds'] > 5.
            return dict(state=self.state, phase=self.phase, manual=True, message=self.message, collected=len(self.frames),
                        window=28, stride=self.stride, busy=self.busy, skipped_windows=self.skipped,
                        frame_id_gaps=self.id_gaps,
                        collected_span_seconds=self.frames[-1][2]-self.frames[0][2] if len(self.frames)>1 else 0.,
                        result=result, pipeline='DSP point cloud -> selected preprocessing -> interpolation to 120 points -> selected classifier')

    def export(self):
        with self.condition:
            if self.saved is None:
                raise ValueError('尚无识别窗口可以保存')
            arrays, report = self.saved
            return {k:v.copy() for k,v in arrays.items()}, copy.deepcopy(report)

    def _worker(self):
        try:
            # Model loading and inference run in the same thread. Torch thread
            # limits are applied there, as required by the existing Nano code.
            preprocess, predictor, labels = self.loader()
            with self.condition:
                self.state, self.message = 'ready', '模型已就绪，点击开始识别后才收集 28 帧'
            while True:
                with self.condition:
                    while self.pending is None and not self.stop:
                        self.condition.wait(.5)
                    if self.stop:
                        return
                    generation, received, frames = self.pending
                    self.pending = None
                    self.busy = True
                begin = self.clock()
                clouds = [f[1] for f in frames]
                points, _, detail = preprocess.run(clouds)
                counts = [int(f['filtered_count']) for f in detail['frames']]
                valid = sum(n >= 3 for n in counts)
                result = dict(type='rejected', reason='too_few_valid_frames', valid_frames=valid,
                              first_frame=frames[0][0], last_frame=frames[-1][0],
                              window_span_seconds=frames[-1][2]-frames[0][2],
                              raw_counts=[len(c) for c in clouds], filtered_counts=counts,
                              input_shape=list(points.shape), _received=received,
                              last_frame_imputed=bool(detail['frames'][-1].get('imputed', False)),
                              upsampled_xyz=points[-1].tolist() if any(counts) else [])
                # Preserve the copied program's gate, after filtering and before
                # classification. Filled empty frames are never counted as observed valid frames.
                if valid >= 8:
                    prediction = predictor.predict(points)
                    category = int(prediction['labels'][0])
                    probability = float(prediction['confidence'][0])
                    if not np.isfinite(probability):
                        raise ValueError('Model returned nonfinite confidence')
                    result.update(type='prediction', reason='', class_id=category,
                                  label=labels[str(category)], confidence=probability,
                                  probabilities=prediction['probabilities'][0].tolist(),
                                  model_name=getattr(predictor, 'runtime_name', predictor.method))
                result['processing_seconds'] = self.clock()-begin
                arrays = dict(points=points,
                              frame_ids=np.asarray([f[0] for f in frames], dtype=np.uint32),
                              raw_xyz=np.concatenate(clouds, axis=0),
                              raw_offsets=np.concatenate(([0], np.cumsum([len(c) for c in clouds]))).astype(np.int32))
                report = dict(result=result, preprocessing=detail)
                with self.condition:
                    self.busy = False
                    if generation == self.generation and not self.stop:
                        self.phase = 'done'
                        self.result = result
                        self.saved = (arrays, report)
                        self.message = '本次识别完成；点击开始识别可再测一次' if valid >= 8 else '本次有效点云不足，未分类；请点击开始重试'
        except Exception as exc:
            with self.condition:
                self._reset('识别出错：' + type(exc).__name__ + ': ' + str(exc))
                self.busy = False
                self.state = 'error'



