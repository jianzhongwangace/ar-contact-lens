"""Local offline video preview; sequential decoding, orientation and media-time tracking."""
import argparse
from collections import deque
import json
import math
from pathlib import Path
import time
import hashlib
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from eye_pipeline import EyePipeline,ROOT,render
from lens_import import LensImportControl
from app_layout import open_window,show,bind_mouse,WIDTH,CONTENT_HEIGHT,HEIGHT,full_canvas,wait_key


def choose_video():
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk();root.withdraw();root.attributes('-topmost',True)
    try:
        selected=filedialog.askopenfilename(parent=root,title='导入离线视频（支持 1080×1920 竖屏）',
            filetypes=[('视频文件','*.mp4 *.mov *.mkv *.avi *.m4v *.webm'),('所有文件','*.*')])
        return Path(selected) if selected else None
    finally:root.destroy()


class VideoReader:
    """Retains just one decoded frame. No camera backend or full-video RAM cache."""
    def __init__(self,path):
        self.path=Path(path).resolve()
        if not self.path.is_file():raise FileNotFoundError(f'视频文件不存在：{self.path}')
        self.cap=cv2.VideoCapture(str(self.path),cv2.CAP_FFMPEG)
        try:
            if not self.cap.isOpened():raise ValueError('无法打开视频。请使用可解码的 MP4/MOV 文件。')
            self.rotation=int(self.cap.get(cv2.CAP_PROP_ORIENTATION_META))%360
            # Desktop OpenCV builds can default this to false. Never rotate twice.
            self.auto_rotation=bool(self.cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1) and
                                    self.cap.get(cv2.CAP_PROP_ORIENTATION_AUTO))
            rate=self.cap.get(cv2.CAP_PROP_FPS)
            self.fps=rate if math.isfinite(rate) and .1<=rate<=240 else 30.
            self.fps_estimated=self.fps!=rate
            count=self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.frame_count=max(0,int(count)) if math.isfinite(count) else 0
            self.duration=self.frame_count/self.fps if self.frame_count else 0.
            self.next_index=0
            self.last_time=None
            first=self.read()
            if first is None:raise ValueError('视频没有可以读取的画面。')
            self.first=first
            self.height,self.width=first[0].shape[:2]
        except Exception:
            self.cap.release();raise

    def _rotate(self,frame):
        if self.auto_rotation:return frame
        rotations={90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}
        return cv2.rotate(frame,rotations[self.rotation]) if self.rotation in rotations else frame

    def read(self):
        ok,frame=self.cap.read()
        if not ok or frame is None:
            if self.frame_count and self.next_index<self.frame_count-2:
                raise RuntimeError(f'视频在第 {self.next_index+1} 帧解码失败，未把它当作正常结束。')
            return None
        stamp=self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        fallback=self.next_index/self.fps
        if not math.isfinite(stamp) or stamp<0 or (self.last_time is not None and stamp<=self.last_time):
            stamp=max(fallback,(self.last_time or 0)+1/self.fps) if self.next_index else 0.
        index=self.next_index;self.next_index+=1;self.last_time=stamp
        return self._rotate(frame),stamp,index

    def seek(self,index):
        index=max(0,min(int(index),max(0,self.frame_count-1)))
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES,index):
            raise RuntimeError('此视频不支持定位，请重新导入后播放。')
        actual=self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.next_index=max(0,int(round(actual)))
        self.last_time=None
        frame=self.read()
        if frame is None:raise RuntimeError('无法读取定位后的画面。')
        return frame

    def close(self):self.cap.release()


class VideoSession:
    """State machine independent of the OpenCV window; seeks invalidate all tracking."""
    def __init__(self,pipeline):
        self.pipeline=pipeline
        self.reader=None
        self.frame=None
        self.eyes=[]
        self.timing={'mesh_ms':0.,'iris_ms':0.,'total_ms':0.}
        self.timestamp=0.;self.index=0
        self.paused=False;self.ended=False
        self.stats={'decoded_frames':0,'frames_with_face':0,'frames_with_lens_pose':0}

    def _accept(self,item):
        frame,stamp,index=item
        eyes,timing=self.pipeline.process(frame,timestamp=stamp)
        self.frame,self.timestamp,self.index=frame,stamp,index
        self.eyes,self.timing=eyes,timing
        self.stats['decoded_frames']+=1
        self.stats['frames_with_face']+=int(bool(eyes))
        self.stats['frames_with_lens_pose']+=int(any(e.lens_pose is not None for e in eyes))

    def load(self,path):
        # Validate new stream and first frame before replacing a working source.
        candidate=VideoReader(path)
        try:
            self.pipeline.reset_tracking()
            self._accept(candidate.first)
        except Exception:
            candidate.close();raise
        old=self.reader;self.reader=candidate
        if old:old.close()
        self.paused=False;self.ended=False
        self.stats={'decoded_frames':1,'frames_with_face':int(bool(self.eyes)),
                    'frames_with_lens_pose':int(any(e.lens_pose is not None for e in self.eyes))}

    def advance(self):
        if self.reader is None or self.paused or self.ended:return False
        item=self.reader.read()
        if item is None:
            self.ended=True;self.paused=True;return False
        self._accept(item);return True

    def seek(self,index):
        if self.reader is None:return
        item=self.reader.seek(index)
        self.pipeline.reset_tracking()
        self._accept(item)
        self.ended=False

    def restart(self):
        self.seek(0);self.paused=False

    def close(self):
        if self.reader:self.reader.close()


class ProcessedVideoSession:
    """Playback-only session for a fully rendered local result video."""
    def __init__(self,path,detail_dir=None):
        self.reader=VideoReader(path)
        self.frame,self.timestamp,self.index=self.reader.first
        self.eyes=[]
        self.detail_dir=Path(detail_dir) if detail_dir else None
        self.detail_images={}
        self.diagnostic_images={}
        self.timing={'mesh_ms':0.,'iris_ms':0.,'total_ms':0.}
        self.paused=False;self.ended=False
        self.processing_complete=True
        self.stats={'decoded_frames':0,'frames_with_face':0,'frames_with_lens_pose':0,'skipped_frames':0}
        self._load_details()

    def _load_details(self):
        self.detail_images={}
        self.diagnostic_images={}
        if not self.detail_dir:return
        for side in ('right','left'):
            path=self.detail_dir/f'{self.index:06d}-{side}.jpg'
            if path.exists():
                image=cv2.imdecode(np.fromfile(str(path),np.uint8),cv2.IMREAD_COLOR)
                if image is not None:self.detail_images[side]=image
            diag=self.detail_dir/f'{self.index:06d}-{side}-diagnostic.jpg'
            if diag.exists():
                image=cv2.imdecode(np.fromfile(str(diag),np.uint8),cv2.IMREAD_COLOR)
                if image is not None:self.diagnostic_images[side]=image

    def advance(self):
        if self.paused or self.ended:return False
        item=self.reader.read()
        if item is None:self.ended=True;self.paused=True;return False
        self.frame,self.timestamp,self.index=item
        self._load_details()
        self.stats['decoded_frames']+=1
        return True

    def seek(self,index):
        self.frame,self.timestamp,self.index=self.reader.seek(index)
        self._load_details()
        self.ended=False

    def restart(self):
        self.seek(0);self.paused=False

    def close(self):self.reader.close()


def render_cached_details(canvas,detail_images,mirror=False,diagnostic_images=None):
    """Replace the empty close-up slots with pre-rendered result crops."""
    for row,side in enumerate(('right','left')):
        image=detail_images.get(side)
        if image is None:continue
        # render() left pane boxes are x=16, y=58+row*270+40, 332×195.
        x,y,w,h=16,98+row*270,332,195
        factor=min(w/image.shape[1],h/image.shape[0])
        resized=cv2.resize(image,(max(1,round(image.shape[1]*factor)),max(1,round(image.shape[0]*factor))))
        if mirror:resized=cv2.flip(resized,1)
        xoff=x+(w-resized.shape[1])//2;yoff=y+(h-resized.shape[0])//2
        canvas[yoff:yoff+resized.shape[0],xoff:xoff+resized.shape[1]]=resized
        canvas[y+h+2:y+h+30,16:360]=24
        cv2.putText(canvas,'Cached try-on result',(18,y+h+22),cv2.FONT_HERSHEY_SIMPLEX,.46,(190,210,220),1,cv2.LINE_AA)
        if diagnostic_images and side in diagnostic_images:
            dy=60+row*230
            canvas[dy:dy+220,1390:1920]=cv2.resize(diagnostic_images[side],(530,220))
    if detail_images:
        canvas[690:734,1390:1920]=24
        cv2.putText(canvas,'Cached inference from the matching source frame',(1395,710),cv2.FONT_HERSHEY_SIMPLEX,.4,(190,210,220),1,cv2.LINE_AA)
    return canvas


def rendered_cache_path(source,control,geometry=None):
    source=Path(source).resolve()
    # Model and renderer changes invalidate prior baked pixels automatically.
    code_hash=hashlib.sha256()
    for relative in ('eye_pipeline.py','lens_renderer.py','center_tracking.py','turn_tracking.py','ellseg_adapter.py',
                     'EllSeg部署/ellseg_core.py','EllSeg部署/weights/ellseg.json'):
        code_hash.update((ROOT/relative).read_bytes())
    material_hash=hashlib.sha256(control.material.rgba.tobytes()).hexdigest() if control.material is not None else 'none'
    geometry=geometry or getattr(control,'geometry_mode','ellseg')
    key=f'{source}:{source.stat().st_size}:{source.stat().st_mtime_ns}:{material_hash}:{control.strength}:{control.lens_scale}:{geometry}:{code_hash.hexdigest()}:details-v2'
    digest=hashlib.sha256(key.encode('utf-8')).hexdigest()[:20]
    cache_root=Path('D:/ARLensRuntime/video-cache') if Path('D:/').exists() else ROOT/'video-cache'
    cache_root.mkdir(parents=True,exist_ok=True)
    return cache_root/f'{source.stem}-{digest}-tryon.mp4'

def rendered_detail_dir(output):
    path=output.with_suffix('')
    return path.parent/(path.name+'-details')


def process_video_to_file(source,pipeline,control,title=None,headless=False):
    """Complete sequential render before playback. Returns an encoded, orientation-normalized file."""
    source_reader=VideoReader(source)
    output=rendered_cache_path(source,control,pipeline.geometry_mode)
    details=rendered_detail_dir(output)
    complete=details/'complete.json'
    if output.exists() and output.stat().st_size>1024 and complete.exists():
        source_reader.close();return output,0,True
    complete.unlink(missing_ok=True)
    writer=None;count=0;faces=0;poses=0
    details.mkdir(parents=True,exist_ok=True)
    try:
        pipeline.reset_tracking()
        first=source_reader.first
        width,height=source_reader.width,source_reader.height
        writer=cv2.VideoWriter(str(output),cv2.VideoWriter_fourcc(*'mp4v'),source_reader.fps,(width,height))
        if not writer.isOpened():raise RuntimeError('无法创建本地处理结果视频。')
        item=first
        while item is not None:
            frame,stamp,index=item
            eyes,_=pipeline.process(frame,timestamp=stamp)
            result=frame.copy()
            if control.material:
                from lens_renderer import compose_lens
                for eye in eyes:compose_lens(result,eye,control.material,control.strength,control.lens_scale)
            writer.write(result);count+=1;faces+=int(bool(eyes));poses+=int(any(e.lens_pose is not None for e in eyes))
            # Store the final, already composited eye crops for playback UI.
            # This avoids a second AI pass while retaining the left/right detail panes.
            for eye in eyes:
                affine=eye.display_affine if eye.display_affine is not None else eye.affine
                closeup=cv2.warpAffine(result,affine,(eye.crop.shape[1],eye.crop.shape[0]),
                                       borderMode=cv2.BORDER_REFLECT_101)
                cv2.imencode('.jpg',closeup,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tofile(
                    str(details/f'{index:06d}-{eye.side}.jpg'))
                from eye_pipeline import eye_overlay
                diagnostic=np.full((220,530,3),24,np.uint8)
                cv2.putText(diagnostic,f'{eye.side}: {eye.geometry_source}',(5,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(235,235,235),1)
                diagnostic[35:155,5:197]=cv2.resize(eye.crop,(192,120))
                diagnostic[35:155,215:407]=cv2.resize(eye_overlay(eye),(192,120))
                cv2.putText(diagnostic,eye.geometry_reason[:58],(5,183),cv2.FONT_HERSHEY_SIMPLEX,.4,(210,210,210),1)
                cv2.imencode('.jpg',diagnostic,[cv2.IMWRITE_JPEG_QUALITY,92])[1].tofile(str(details/f'{index:06d}-{eye.side}-diagnostic.jpg'))
            if not headless and title:
                progress=(index+1)/source_reader.frame_count if source_reader.frame_count else 0
                canvas=np.full((CONTENT_HEIGHT,WIDTH,3),24,np.uint8)
                cv2.putText(canvas,'Processing video before playback...', (30,60),cv2.FONT_HERSHEY_SIMPLEX,1,(235,235,235),2)
                cv2.putText(canvas,f'{source_reader.path.name}  {index+1}/{source_reader.frame_count or "?"} frames  {progress:.0%}',
                            (30,105),cv2.FONT_HERSHEY_SIMPLEX,.7,(200,215,235),1,cv2.LINE_AA)
                cv2.putText(canvas,'Q / Esc: cancel', (30,145),cv2.FONT_HERSHEY_SIMPLEX,.6,(210,210,210),1,cv2.LINE_AA)
                show(title,canvas)
                if wait_key(title,1) in (27,ord('q')):raise KeyboardInterrupt
            item=source_reader.read()
        writer.release();writer=None;source_reader.close()
        complete.write_text(json.dumps({'frames':count,'geometry':pipeline.geometry_mode,'faces':faces,'lens_poses':poses}),encoding='utf-8')
        return output,count,False
    except BaseException:
        if writer:writer.release()
        source_reader.close()
        try:output.unlink(missing_ok=True)
        except OSError:pass
        raise


class VideoControls:
    HEIGHT=124
    def __init__(self):
        self.font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
        self.pending=None;self.rects={}
        self.cached_key=None;self.panel=None

    def on_mouse(self,event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONUP:return False
        for action,(x0,y0,x1,y1) in self.rects.items():
            if x0<=x<x1 and y0<=y<y1:
                self.pending=('seek',(x-x0)/max(1,x1-x0)) if action=='seek' else (action,None)
                return True
        return False

    def decorate(self,canvas,session,overlay,processing_fps,notice=''):
        h,w=HEIGHT-self.HEIGHT,WIDTH
        r=session.reader
        name=r.path.name if r else '尚未选择视频'
        if len(name)>44:name=name[:41]+'…'
        if getattr(session,'processing_complete',False):
            state='播放结束' if session.ended else ('已暂停' if session.paused else '播放中')
        else:
            state='处理中'
        progress=(session.index+1)/r.frame_count if r and r.frame_count else 0
        info=f'{name}　{r.width}×{r.height}　{session.timestamp:.1f} / {r.duration:.1f} 秒　{state}' if r else name
        key=(w,info,round(progress,3),overlay,int(processing_fps),notice,bool(r))
        if key!=self.cached_key:
            im=Image.new('RGB',(w,self.HEIGHT),(32,38,46));draw=ImageDraw.Draw(im)
            buttons=[('换视频',20,170),('重新播放' if session.ended else ('播放' if session.paused else '暂停'),190,320),
                     ('从头播放',340,490),('原眼对比' if overlay else '查看试戴',510,670),('保存当前帧',690,860)]
            for text,x0,x1 in buttons:
                draw.rounded_rectangle((x0,10,x1,46),radius=6,fill=(65,88,133))
                draw.text((x0+15,17),text,font=self.font,fill='white')
            draw.text((880,17),'空格 播放/暂停 · R 重播 · O 对比 · Q 返回',font=self.font,fill=(225,225,225))
            draw.text((20,54),info,font=self.font,fill=(230,230,230))
            draw.rounded_rectangle((20,85,w-20,95),radius=4,fill=(70,75,83))
            draw.rectangle((20,85,20+int((w-40)*min(1,progress)),95),fill=(95,157,230))
            detail=notice or ('已完成预处理 · 按原视频速度播放 · 无声预览'
                              if getattr(session,'processing_complete',False)
                              else '正在完整处理视频，完成后按原视频速度播放 · 无声预览')
            draw.text((20,100),detail,font=self.font,fill=(190,205,225))
            self.panel=cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR);self.cached_key=key
        self.rects={name:(x0,h+10,x1,h+47) for name,x0,x1 in
                    [('open',20,170),('toggle',190,320),('restart',340,490),('compare',510,670),('save',690,860)]}
        if r and r.frame_count:self.rects['seek']=(20,h+79,w-20,h+100)
        output=full_canvas(canvas).copy()
        output[h:]=self.panel
        return output


def show_video_error(exc):
    import tkinter as tk
    from tkinter import messagebox
    root=tk.Tk();root.withdraw();root.attributes('-topmost',True)
    try:messagebox.showerror('视频无法处理',str(exc),parent=root)
    finally:root.destroy()


def play_video(path=None,lens_control=None,device='auto',headless=False,report=None,max_frames=0):
    """Complete the AI render first, then play the encoded result at source FPS."""
    title='AR eyes | Offline video'
    pipeline=None;session=None
    control=lens_control or LensImportControl()
    previous_callback=control.video_callback
    previous_geometry_callback=control.geometry_callback
    video_ui=VideoControls()
    control.video_callback=lambda selected:setattr(video_ui,'pending',('load',selected))
    summary={'status':'starting','audio_preview':False,'playback_mode':'preprocess_then_realtime'}
    start=time.perf_counter();notice='';last_canvas=None
    try:
        if not headless:
            open_window(title)
            placeholder=np.full((CONTENT_HEIGHT,WIDTH,3),24,np.uint8)
            cv2.putText(placeholder,'Loading video models...', (35,65),cv2.FONT_HERSHEY_SIMPLEX,1,(230,230,230),2)
            show(title,placeholder);wait_key(title,1)
        pipeline=EyePipeline(device=device,geometry=control.geometry_mode)
        summary['device']=str(pipeline.predictor.device)
        summary['geometry_mode']=pipeline.geometry_mode
        current_source=None
        def change_geometry(mode):
            nonlocal current_source
            pipeline.set_geometry(mode)
            if current_source is not None:video_ui.pending=('load',current_source)
        control.geometry_callback=change_geometry
        overlay=True;mirror=False;dirty=True
        def mouse(event,x,y,flags,param):
            if not video_ui.on_mouse(event,x,y,flags,param):control.on_mouse(event,x,y,flags,param)
        if not headless:bind_mouse(title,mouse)

        def load_and_process(source):
            nonlocal session,dirty,notice,current_source
            output,count,cache_hit=process_video_to_file(source,pipeline,control,title,headless)
            if session:session.close()
            session=ProcessedVideoSession(output,rendered_detail_dir(output))
            current_source=Path(source)
            session.stats['processed_frames']=count
            session.stats['cache_hit']=cache_hit
            session.paused=False;session.ended=False
            notice='缓存结果已生成，开始按原视频速度播放' if not cache_hit else '已使用本地缓存结果，按原视频速度播放'
            dirty=True
            return output,count,cache_hit

        if path:
            load_and_process(path)
        if headless:
            if session:
                summary.update(session.stats)
                summary.update(source=str(session.reader.path),display_size=[session.reader.width,session.reader.height],
                               source_fps=session.reader.fps,source_frame_count=session.reader.frame_count,
                               rotation_degrees=session.reader.rotation,last_index=session.index,end_reached=session.ended)
            summary['status']='ok';return summary

        next_play_time=time.perf_counter()
        recent=deque(maxlen=30)
        while True:
            if session is not None and session.frame is not None and dirty:
                fps=(len(recent)-1)/(recent[-1]-recent[0]) if len(recent)>1 else 0
                # Result video already contains the lens. Do not run AI/render again.
                view=render(session.frame,[],session.timing,fps,mirror,False,video_mode=True)
                view=render_cached_details(view,session.detail_images,mirror,session.diagnostic_images)
                view=control.decorate(view)
                last_canvas=video_ui.decorate(view,session,overlay,fps,notice)
                if not session.paused and not session.ended:
                    notice=''
                if not headless:show(title,last_canvas)
                dirty=False
            elif session is None and dirty:
                empty=np.full((CONTENT_HEIGHT,WIDTH,3),24,np.uint8)
                blank=SimpleNamespace(reader=None,paused=True,ended=False,index=0,timestamp=0.,processing_complete=False)
                last_canvas=video_ui.decorate(control.decorate(empty),blank,overlay,0,'点击“换视频”选择文件')
                if not headless:show(title,last_canvas)
                dirty=False
            key=wait_key(title,5)
            if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            action=video_ui.pending;video_ui.pending=None
            mapping={ord(' '):'toggle',ord('r'):'restart',ord('o'):'compare',ord('s'):'save',ord('v'):'open'}
            if key in mapping:action=(mapping[key],None)
            if key==ord('m'):mirror=not mirror;dirty=True
            try:
                if action:
                    kind,value=action
                    if kind=='open':
                        value=choose_video()
                        if value:kind='load'
                    if kind=='load':
                        load_and_process(value);next_play_time=time.perf_counter();recent.clear()
                    elif kind=='toggle' and session:
                        if session.ended:session.restart();next_play_time=time.perf_counter()
                        else:session.paused=not session.paused
                        dirty=True
                    elif kind=='restart' and session:
                        session.restart();next_play_time=time.perf_counter();dirty=True
                    elif kind=='seek' and session:
                        session.seek(round(value*(session.reader.frame_count-1)))
                        next_play_time=time.perf_counter();dirty=True
                    elif kind=='compare':overlay=not overlay;dirty=True
                    elif kind=='save' and session and session.frame is not None:
                        from image_app import save_result
                        folder=save_result(session.frame,[],last_canvas,control)
                        notice='当前帧已保存到 '+folder.name;dirty=True
                if control.handle_pending(None if key==ord('v') else key):dirty=True
                if session and not session.paused and not session.ended and time.perf_counter()>=next_play_time:
                    advanced=session.advance()
                    if advanced:
                        recent.append(time.perf_counter())
                        next_play_time=time.perf_counter()+max(.001,1/session.reader.fps)
                    dirty=True
            except Exception as exc:
                if session:session.paused=True
                notice=str(exc);dirty=True;show_video_error(exc)
        summary['status']='ok';return summary
    except Exception as exc:
        summary['status']='error';summary['error']=str(exc)
        if headless:raise
        show_video_error(exc);return summary
    finally:
        control.video_callback=previous_callback
        control.geometry_callback=previous_geometry_callback
        if previous_geometry_callback:previous_geometry_callback(control.geometry_mode)
        if session:
            summary.update(session.stats)
            if session.reader:
                r=session.reader
                summary.update(source=str(r.path),display_size=[r.width,r.height],source_fps=r.fps,
                               source_frame_count=r.frame_count,rotation_degrees=r.rotation,
                               last_index=session.index,end_reached=session.ended)
            session.close()
        if pipeline:pipeline.close()
        if not headless:
            try:cv2.destroyWindow(title)
            except cv2.error:pass
        summary['elapsed_seconds']=time.perf_counter()-start
        if report:
            report=Path(report);report.parent.mkdir(parents=True,exist_ok=True)
            report.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser(description='Offline video lens try-on; preserves portrait orientation')
    p.add_argument('--video',type=Path)
    p.add_argument('--lens',type=Path)
    p.add_argument('--device',choices=['auto','cpu','cuda','dml','onnx-cpu'],default='auto')
    p.add_argument('--geometry',choices=['ellseg','facemesh'],default='ellseg')
    p.add_argument('--headless',action='store_true')
    p.add_argument('--frames',type=int,default=0)
    p.add_argument('--every-frame',action='store_true',help='debug: process and display sequential frames (slow)')
    p.add_argument('--report',type=Path)
    a=p.parse_args()
    if a.headless and not a.video:p.error('--headless requires --video PATH')
    control=LensImportControl()
    control.geometry_mode=a.geometry
    if a.lens:control.load(a.lens,persist=not a.headless)
    result=play_video(a.video,control,a.device,a.headless,a.report,a.frames)
    return 0 if result['status']=='ok' else 1

if __name__=='__main__':raise SystemExit(main())
