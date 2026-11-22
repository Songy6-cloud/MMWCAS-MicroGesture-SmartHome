"""One board-triggered acquisition, 28 consecutive samples in <=2 seconds."""
import time
import threading
import numpy as np
from engine_base import Engine as BaseEngine

class Engine(BaseEngine):
    def __init__(self,loader,clock=time.monotonic,retry_delay=.8,max_retries=3):
        super().__init__(loader,clock=clock)
        self.trigger=None
        self.session=int(time.time()*1000)&0xffffffff
        self.captured=0
        self.capture_span=0.
        self.progress_time=self.clock()
        self.capture_bounds=None
        self.last_board_session=None
        self.last_board_index=0
        self.session_mismatches=0
        self.retry_delay=float(retry_delay)
        self.max_retries=int(max_retries)
        self.retry_count=0
        self.retry_reason=''

    def request_capture(self):
        with self.condition:
            if self.stop or self.state!='ready':raise ValueError('模型尚未就绪')
            if self.phase not in ('idle','done') or self.busy:raise ValueError('本次采集/处理尚未结束')
            if self.trigger is None:raise ValueError('板端连接未就绪')
            self._reset('正在请求板端开始连续采集，请准备做手势')
            self.retry_count=0;self.retry_reason=''
            self.session=(self.session+1)&0xffffffff or 1
            self.phase='arming';self.captured=0;self.capture_span=0.;self.capture_bounds=None
            self.progress_time=self.clock();session=self.session
        try:self.trigger(session)
        except Exception:
            with self.condition:self._reset('触发失败，请检查连接后重试')
            raise

    def _prepare_retry_locked(self,reason,wait_for_connection=False):
        if self.retry_count>=self.max_retries:
            self._reset('连续 %d 次未收满 28 帧：%s；请检查雷达连接后重新开始'%(self.max_retries,reason))
            self.captured=0
            return
        self.retry_count+=1;self.retry_reason=reason
        self.generation+=1;self.frames.clear();self.pending=self.result=self.saved=None
        self.captured=0;self.capture_span=0.;self.capture_bounds=None
        self.session=(self.session+1)&0xffffffff or 1
        self.progress_time=self.clock()
        self.phase='reconnecting' if wait_for_connection else 'retrying'
        self.message='只收到部分帧，正在自动重试 %d/%d（不会用不完整批次识别）'%(self.retry_count,self.max_retries)
        if not wait_for_connection:
            timer=threading.Timer(self.retry_delay,self._retry_trigger,args=(self.session,))
            timer.daemon=True;timer.start()

    def _retry_trigger(self,session):
        with self.condition:
            if self.stop or self.phase!='retrying' or session!=self.session:return
            self.phase='arming';self.progress_time=self.clock()
        try:self.trigger(session)
        except Exception:
            with self.condition:
                if not self.stop and session==self.session and self.phase=='arming':
                    self.phase='reconnecting'
                    self.message='自动重试等待雷达重新连接'

    def connection_ready(self):
        with self.condition:
            if self.stop or self.phase!='reconnecting':return
            self.phase='retrying';session=self.session
            timer=threading.Timer(self.retry_delay,self._retry_trigger,args=(session,))
            timer.daemon=True;timer.start()

    def disconnected(self,error):
        with self.condition:
            # A complete 28-frame recognition no longer depends on the radar
            # socket. Earlier disconnects retry the whole burst instead of
            # accepting a two-frame (or otherwise partial) batch.
            if self.phase in ('arming','capturing','receiving','retrying'):
                self._prepare_retry_locked('雷达连接短暂中断',wait_for_connection=True)

    def offer(self,frame,now=None):
        now=self.clock() if now is None else now
        with self.condition:
            if self.stop or self.phase not in ('arming','capturing','receiving'):return
            b=frame.get('burst')
            if not b:return
            self.last_board_session=b['session'];self.last_board_index=b['index']
            if b['session']!=self.session:
                self.session_mismatches+=1
                self.message='板端批次号不匹配：等待 %u，收到 %u'%(self.session,b['session'])
                return
            self.progress_time=now
            if b['error'] or b['phase']==4:
                self._prepare_retry_locked('板端采集不连续或时间戳不合格（代码 %s）'%b['error'])
                return
            self.captured=b['captured']
            self.capture_span=(b['last_us']-b['first_us'])/1e6 if self.captured else 0.
            if not 0<=self.capture_span<=2.:
                self._prepare_retry_locked('采集窗口超过 2 秒');return
            if self.captured<28:
                self.phase='capturing';self.message='请做手势：板端已采集 %d/28 帧'%self.captured;return
            self.phase='receiving'
            ordinal=b['index']
            if not ordinal:
                self.message='2 秒内采集完成，等待 DSP 按原算法处理';return
            rd=frame.get('rd')
            if (ordinal!=len(self.frames)+1 or not rd or rd.get('version')!=2 or rd.get('selection_enabled')!=2):
                self._prepare_retry_locked('DSP 处理结果顺序中断或固件不匹配');return
            bounds=(b['first_us'],b['last_us'])
            if self.capture_bounds is None:self.capture_bounds=bounds
            if bounds!=self.capture_bounds or not bounds[0]<=b['sample_us']<=bounds[1]:
                self._prepare_retry_locked('原始采集时间戳不一致');return
            if ordinal==1 and b['sample_us']!=bounds[0]:
                self._prepare_retry_locked('缺少采集首帧');return
            if ordinal==28 and b['sample_us']!=bounds[1]:
                self._prepare_retry_locked('缺少采集末帧');return
            stamp=b['sample_us']/1e6
            if self.frames:
                if ((frame['frame_id']-self.frames[-1][0])%2**32!=1 or not .049999<=stamp-self.frames[-1][2]<=.090001):
                    self._prepare_retry_locked('采集帧不连续');return
            xyz=np.asarray(frame['xyz'],dtype=np.float64).reshape(-1,3).copy()
            if not np.isfinite(xyz).all():self._prepare_retry_locked('收到无效点云');return
            self.frames.append((frame['frame_id'],xyz,stamp))
            self.message='采集完成，DSP 已处理 %d/28 帧'%len(self.frames)
            if ordinal==28:
                self.phase='processing';self.message='28 帧已处理完成，正在插值和识别'
                self.pending=(self.generation,now,list(self.frames));self.condition.notify()

    def snapshot(self):
        with self.condition:
            idle=self.clock()-self.progress_time
            if self.phase in ('arming','capturing') and idle>30:
                self._reset('等待板端采集响应超时，请检查诊断日志后重试')
            elif self.phase in ('retrying','reconnecting') and idle>30:
                self._reset('自动重试等待连接超过 30 秒，请检查雷达连接后重试')
            elif self.phase=='receiving' and idle>300:
                self._reset('等待DSP批处理结果超过5分钟，请检查接收错误后重试')
        out=super().snapshot()
        out.update(collected=self.captured,processed=len(self.frames),capture_span_seconds=self.capture_span,
                   window=28,acquisition_limit_seconds=2.,session=self.session,
                   last_board_session=self.last_board_session,last_board_index=self.last_board_index,
                   session_mismatches=self.session_mismatches,
                   auto_retry_count=self.retry_count,auto_retry_limit=self.max_retries,
                   auto_retry_reason=self.retry_reason,
                   progress_source='board_burst_header',
                   progress_session_verified=(self.last_board_session==self.session))
        return out


