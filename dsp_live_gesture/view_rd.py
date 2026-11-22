"""Display actual DSP RD decisions. No host clustering or angular filtering."""
import argparse
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import socket
from socketserver import ThreadingMixIn
import threading
import time
from urllib.parse import urlparse, parse_qs
from dsp_pointcloud import receive_frame

BASE=Path(__file__).resolve().parent

class State:
    def __init__(self, output, mode):
        self.lock=threading.Lock();self.output=Path(output);self.mode=mode
        self.frames=deque(maxlen=28);self.times=deque(maxlen=512);self.rate_start=None;self.revision=0
        self.last=None;self.connected=False;self.error='';self.previous=None;self.gaps=0
    def accept(self,f):
        now=time.monotonic();key=(f.get('channel',0),f['dsp_id'])
        with self.lock:
            if self.previous and key!=self.previous[1]:
                self.times.clear();self.rate_start=None
            delta=(f['frame_id']-self.previous[0])%2**32 if self.previous else 1
            if self.previous and key==self.previous[1] and delta==0:return
            if self.previous and (key!=self.previous[1] or delta>=2**31 or (self.last is not None and now-self.last>2.)):
                self.frames.clear();self.gaps+=1
            elif delta>1:self.gaps+=1
            # Forward frame-number gaps do not invent frames or erase received history.
            # Keep heatmap only for the newest frame; last 28 detections stay bounded.
            if self.frames and self.frames[-1].get('rd'):
                self.frames[-1]['rd'].pop('power',None)
            self.frames.append(f);self.times.append(now);self.last=now;self.revision+=1
            if self.rate_start is None:self.rate_start=now
            self.previous=(f['frame_id'],key);self.connected=True;self.error=''
    def disconnected(self,e):
        with self.lock:
            self.connected=False;self.error=str(e);self.frames.clear();self.times.clear();self.last=None;self.previous=None;self.rate_start=None;self.revision+=1
    def snapshot(self, revision=None):
        with self.lock:
            now=time.monotonic()
            while self.times and self.times[0]<=now-5.:self.times.popleft()
            elapsed=min(5.,max(0.,now-self.rate_start)) if self.rate_start is not None else 0.
            boundary=max(now-5.,self.rate_start) if self.rate_start is not None else now
            fps=sum(t>boundary for t in self.times)/elapsed if elapsed>=.5 else 0.
            unchanged=revision is not None and revision==self.revision
            f=self.frames[-1] if self.frames and not unchanged else None
            frame=dict(f,rd=dict(f['rd']) if f.get('rd') else None) if f else None
            return dict(mode=self.mode,connected=self.connected,error=self.error,frame=frame,gaps=self.gaps,
                age=now-self.last if self.last is not None else None,fps=fps,
                fps_window_s=elapsed,revision=self.revision,unchanged=unchanged)
    def save(self):
        with self.lock:
            if not self.frames:raise ValueError('No frames received')
            # Serialize while locked: the reader retires old heatmaps in place.
            data=json.dumps(dict(format='mmwcas-dsp-rd-v2',mode=self.mode,frames=list(self.frames)),ensure_ascii=True)
        self.output.mkdir(parents=True,exist_ok=True)
        path=self.output/('rd-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f')+'.json')
        path.write_text(data,encoding='utf-8');return str(path.resolve())

class Server(ThreadingMixIn,HTTPServer):
    daemon_threads=True
    allow_reuse_address=True
class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def respond(self,status,data,kind='application/json; charset=utf-8'):
        raw=data if isinstance(data,bytes) else json.dumps(data,ensure_ascii=True).encode('utf-8')
        self.send_response(status);self.send_header('Content-Type',kind)
        self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers()
        try:self.wfile.write(raw)
        except (BrokenPipeError,ConnectionResetError):pass
    def do_GET(self):
        path=urlparse(self.path).path
        if path=='/':self.respond(200,(BASE/'rd_view.html').read_bytes(),'text/html; charset=utf-8')
        elif path=='/api/state':
            query=parse_qs(urlparse(self.path).query)
            try:revision=int(query.get('revision',[''])[0])
            except ValueError:revision=None
            value=self.server.state.snapshot(revision)
            # The compact Smart Room page needs recognition state only. Do not
            # serialize the large RD heatmap/point-cloud frame on every poll.
            if query.get('compact',['0'])[0]=='1':
                value.pop('frame',None)
                # Once the browser has received a result, subsequent idle
                # heartbeats do not resend its probability/point arrays.
                if query.get('result',['1'])[0]=='0' and value.get('recognition'):
                    value['recognition']=dict(value['recognition'])
                    value['recognition'].pop('result',None)
            self.respond(200,value)
        else:self.respond(404,{'error':'Not found'})
    def do_POST(self):
        hosts=('127.0.0.1:'+str(self.server.server_port),'localhost:'+str(self.server.server_port))
        origin=self.headers.get('Origin')
        if self.headers.get('Host') not in hosts or (origin and origin not in ['http://'+h for h in hosts]):
            self.respond(403,{'error':'Local origin required'});return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if self.path!='/api/save' or not 0<length<=128 or self.headers.get('Content-Type')!='application/json':
                raise ValueError('Expected small JSON save request')
            json.loads(self.rfile.read(length).decode('utf-8'))
            self.respond(200,{'path':self.server.state.save()})
        except (ValueError,OSError) as e:self.respond(400,{'error':str(e)})

def demo_frame(n):
    # UI fixture only; unmistakably labelled DEMO. Never presented as real capture.
    nr=nd=128;cells=[];power=[1.]* (nr*nd);xyz=[];polar=[];bins=[]
    for center,flags in [(3,1),(14,7),(29,3),(50,1)]:
        for r in range(center-1,center+2):
            for d in range(-2,3):
                cells.append([r,d,flags]);power[(d%nd)*nr+r-1]=1000.*math.exp(-.5*((r-center)**2+d*d))
                if flags==7:
                    rr=(r-1)*.037734;az=.1*math.sin(n*.1)+d*.015
                    xyz.append([rr*math.sin(az),rr*math.cos(az),(r-center)*.02]);polar.append([rr,d*.34,az,0.,10.]);bins.append([r-1,d])
    rd=dict(version=1,range_bins=nr,doppler_bins=nd,min_bin=6,max_bin=40,min_area=3,raw_cells=60,in_range_cells=30,
        components=4,valid_components=2,selected_cells=15,aoa_input_cells=15,selected_min_bin=13,selected_max_bin=15,
        cap_dropped=0,selection_enabled=1,k0=6.3,kd=6.3,cells=cells,power=power)
    return dict(frame_id=n,dsp_id=0,channel=0,xyz=xyz,polar=polar,bins=bins,rd=rd,
                system=dict(range_resolution_m=.037734,doppler_resolution_m_s=.34,range_bins=nr,doppler_bins=nd))

def reader(state,stop,host,port,demo=False,replay=None):
    if demo or replay:
        i=0
        while not stop.is_set():
            # Replay is copied because State retires previous heatmaps.
            f=demo_frame(i) if demo else json.loads(json.dumps(replay[i%len(replay)]))
            state.accept(f);i+=1;stop.wait(.07)
        return
    while not stop.is_set():
        try:
            with socket.create_connection((host,port),timeout=3.) as sock:
                sock.settimeout(30.)
                if hasattr(state,'attach_socket'):state.attach_socket(sock)
                while not stop.is_set():state.accept(receive_frame(sock,include_rd=True))
        except (OSError,ValueError,EOFError) as e:
            state.disconnected(e);stop.wait(2.)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',default='192.168.33.180');p.add_argument('--port',type=int,default=29172)
    p.add_argument('--http-port',type=int,default=8765);p.add_argument('--output',default=str(BASE/'dsp-rd-snapshots'))
    group=p.add_mutually_exclusive_group();group.add_argument('--demo',action='store_true');group.add_argument('--replay')
    a=p.parse_args();replay=None
    if a.replay:
        path=Path(a.replay)
        if path.stat().st_size>32*1024*1024:p.error('Snapshot exceeds 32 MiB')
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('format')!='mmwcas-dsp-rd-v2' or not 1<=len(data.get('frames',[]))<=28:p.error('Unsupported snapshot')
        replay=data['frames']
    state=State(a.output,'DEMO / synthetic' if a.demo else ('REPLAY' if replay else 'LIVE / DSP'))
    server=Server(('127.0.0.1',a.http_port),Handler);server.state=state
    stop=threading.Event();thread=threading.Thread(target=reader,args=(state,stop,a.host,a.port,a.demo,replay),daemon=True);thread.start()
    print('Open on THIS computer: http://127.0.0.1:%d' % server.server_port,flush=True)
    print('Mode: '+state.mode+' | DSP selects; this viewer never selects/clusters. Ctrl+C to stop.',flush=True)
    try:server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:pass
    finally:stop.set();server.server_close();thread.join(4.)
if __name__=='__main__':main()
