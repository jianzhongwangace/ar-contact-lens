"""USB webcam demo. Run using run_camera.cmd or .venv/Scripts/python.exe."""
import argparse
from collections import deque
import json
import os
from pathlib import Path
import threading
import time
import traceback

import cv2
import numpy as np

from eye_pipeline import EyePipeline, DEFAULT_WEIGHTS, ROOT, render
from lens_import import LensImportControl
from app_layout import open_window,show,bind_mouse,wait_key


def open_camera(index, backend, width, height):
    api = {'dshow':cv2.CAP_DSHOW,'msmf':cv2.CAP_MSMF,'auto':cv2.CAP_ANY}[backend]
    cap = cv2.VideoCapture(index,api)
    if not cap.isOpened():
        cap.release()
        return None,None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    cap.set(cv2.CAP_PROP_FPS,30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
    # DirectShow can reset the format to YUY2 when resolution/FPS are changed.
    # Negotiate MJPG LAST: this webcam only delivers YUY2 1080p at 5 FPS.
    cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
    for _ in range(15):
        ok,frame = cap.read()
        if ok and frame is not None and frame.size:
            return cap,frame
    cap.release()
    return None,None


class LatestCamera:
    """Drain capture continuously; only the most recent complete frame is retained."""
    def __init__(self,cap,first_frame):
        self.cap=cap
        self.lock=threading.Lock()
        self.frame=first_frame
        self.sequence=0
        self.timestamp=time.perf_counter()
        self.stopped=threading.Event()
        self.failed=False
        self.worker=threading.Thread(target=self._read,name='camera-reader',daemon=True)
        self.worker.start()

    def _read(self):
        failures=0
        try:
            while not self.stopped.is_set():
                ok,frame=self.cap.read()
                if not ok or frame is None:
                    failures+=1
                    if failures>=15:
                        self.failed=True
                        break
                    self.stopped.wait(.03)
                    continue
                failures=0
                with self.lock:
                    self.frame=frame
                    self.sequence+=1
                    self.timestamp=time.perf_counter()
        finally:
            self.cap.release()

    def get(self):
        with self.lock:
            return self.sequence,self.timestamp,self.frame.copy()

    def close(self):
        self.stopped.set()
        self.worker.join(timeout=3)


def save_snapshot(canvas,eyes,frame=None,lens_control=None):
    folder=ROOT/'output'/time.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True,exist_ok=True)
    cv2.imencode('.png',canvas)[1].tofile(str(folder/'preview.png'))
    if frame is not None and lens_control and lens_control.material:
        from lens_renderer import compose_lens
        result=frame.copy()
        for eye in eyes:compose_lens(result,eye,lens_control.material,lens_control.strength,lens_control.lens_scale)
        cv2.imencode('.png',result)[1].tofile(str(folder/'result.png'))
    for eye in eyes:
        for name,mask in [('visible_iris',eye.iris),('pupil',eye.pupil),('eyelid',eye.eyelid)]:
            cv2.imencode('.png',mask*255)[1].tofile(str(folder/f'{eye.side}_{name}.png'))
    print('Snapshot saved locally:',folder,flush=True)


def main():
    parser=argparse.ArgumentParser(description='USB camera: FaceMesh + supplied LightIrisNet checkpoint')
    parser.add_argument('--camera',default='auto',help='auto or numeric index such as 0/1/2')
    parser.add_argument('--backend',choices=['dshow','msmf','auto'],default='dshow')
    parser.add_argument('--list-cameras',action='store_true')
    parser.add_argument('--max-cameras',type=int,default=4)
    parser.add_argument('--width',type=int,default=1280)
    parser.add_argument('--height',type=int,default=720)
    parser.add_argument('--device',choices=['auto','cpu','cuda','dml','onnx-cpu'],default='auto')
    parser.add_argument('--geometry',choices=['ellseg','facemesh'],default='ellseg')
    parser.add_argument('--threads',type=int,default=min(4,os.cpu_count() or 1))
    parser.add_argument('--input-width',type=int,default=256)
    parser.add_argument('--input-height',type=int,default=160)
    parser.add_argument('--threshold',type=float,default=.5)
    parser.add_argument('--close-threshold',type=float,default=.10)
    parser.add_argument('--weights',type=Path,default=DEFAULT_WEIGHTS)
    parser.add_argument('--lens',type=Path,help='Lens product photo or RGBA texture')
    parser.add_argument('--no-mirror',action='store_true')
    parser.add_argument('--headless',action='store_true',help='No window; useful for camera validation')
    parser.add_argument('--frames',type=int,default=0,help='Stop after N processed frames; 0 is unlimited')
    parser.add_argument('--seconds',type=float,default=0,help='Stop after N seconds of capture')
    parser.add_argument('--image',type=Path,help='Process a local test image instead of camera')
    parser.add_argument('--report',type=Path,help='Write diagnostic numbers only; no images')
    args=parser.parse_args()
    if not 0<args.threshold<1:
        parser.error('--threshold must be between 0 and 1')
    if args.headless and not args.frames and not args.seconds and not args.image and not args.list_cameras:
        parser.error('--headless requires --frames N or --seconds N')
    if args.list_cameras:
        found=[]
        for i in range(args.max_cameras):
            cap,frame=open_camera(i,args.backend,args.width,args.height)
            if cap is not None:
                print(f'Camera {i}: OK {frame.shape[1]}x{frame.shape[0]} ({args.backend})',flush=True)
                found.append(i)
                cap.release()
            else:
                print(f'Camera {i}: unavailable',flush=True)
        print('Available camera indices:',found)
        return 0 if found else 2

    pipeline=None
    camera=None
    cap=None
    report={'status':'starting','processed_frames':0,'frames_with_face':0,'frames_with_visible_iris':0,'frames_with_lens_pose':0}
    timings=deque(maxlen=1000)
    eye_diagnostics=deque(maxlen=2000)
    run_started=None
    try:
        print('Loading FaceMesh and verified LightIrisNet weights...',flush=True)
        pipeline=EyePipeline(args.weights,args.device,args.input_width,args.input_height,
                             args.threshold,args.close_threshold,args.threads,static_image=bool(args.image),geometry=args.geometry)
        report['geometry_mode']=pipeline.geometry_mode
        report['device']=str(pipeline.predictor.device)
        if getattr(pipeline.predictor,'adapter',None):
            report['gpu_adapter']=pipeline.predictor.adapter
        report['parameters']=pipeline.predictor.parameters
        print(f'Loaded {report["parameters"]:,} parameters on {report["device"]}.',flush=True)
        if args.image:
            frame=cv2.imdecode(np.fromfile(str(args.image),dtype=np.uint8),cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError('Cannot decode input image')
        else:
            indices=range(args.max_cameras) if args.camera=='auto' else [int(args.camera)]
            for i in indices:
                print(f'Opening camera {i} with {args.backend}...',flush=True)
                cap,frame=open_camera(i,args.backend,args.width,args.height)
                if cap is not None:
                    report['camera_index']=i
                    report['capture_size']=[frame.shape[1],frame.shape[0]]
                    code=int(cap.get(cv2.CAP_PROP_FOURCC))
                    report['capture_fourcc']=''.join(chr((code>>(8*j))&255) for j in range(4))
                    report['camera_reported_fps']=cap.get(cv2.CAP_PROP_FPS)
                    print(f'Capture {report["capture_size"]} {report["capture_fourcc"]}; reported FPS={report["camera_reported_fps"]:.1f}',flush=True)
                    break
            if cap is None:
                raise RuntimeError('No readable USB camera. Connect it, close apps using it, enable Windows camera access, '
                                   'then run run_camera.cmd --list-cameras; try --backend msmf if needed.')
            camera=LatestCamera(cap,frame)
            cap=None  # Ownership belongs to the capture worker.
        title='AR eyes | FaceMesh + LightIrisNet'
        lens_control=LensImportControl() if not args.headless or args.lens else None
        if lens_control:
            lens_control.geometry_mode=pipeline.geometry_mode
            lens_control.geometry_callback=pipeline.set_geometry
        if args.lens:lens_control.load(args.lens,persist=not args.headless)
        if not args.headless:
            open_window(title)
            bind_mouse(title,lens_control.on_mouse)
        previous_sequence=-1
        times=deque(maxlen=30)
        mirror=not args.no_mirror
        overlay=True
        run_started=time.perf_counter()
        while True:
            loop_start=time.perf_counter()
            if args.seconds and loop_start-run_started>=args.seconds:
                break
            if camera:
                if camera.failed:
                    raise RuntimeError('Camera disconnected or stopped delivering frames.')
                sequence,received_at,frame=camera.get()
                if loop_start-received_at>5:
                    raise RuntimeError('Camera timed out: no fresh frame for 5 seconds.')
                if sequence==previous_sequence:
                    if not args.headless and wait_key(title,5) in (27,ord('q')):
                        break
                    time.sleep(.005)
                    continue
                previous_sequence=sequence
            eyes,timing=pipeline.process(frame,timestamp=received_at if camera else None)
            timing['has_eyes']=bool(eyes)
            timings.append(timing)
            for eye in eyes:
                within=eye.eyelid>0
                eye_diagnostics.append({
                    'open':eye.is_open,'opening':eye.openness,
                    'source_eye_width':float((pipeline.size[0]-1)/np.linalg.norm(eye.affine[0,:2])/1.7),
                    'raw_iris_max':float(eye.iris_prob.max()),
                    'inside_iris_max':float(eye.iris_prob[within].max()) if within.any() else 0.,
                    'visible_pixels':int(eye.iris.sum()),
                })
            report['processed_frames']+=1
            report['frames_with_face']+=int(bool(eyes))
            report['frames_with_visible_iris']+=int(any(e.iris.any() for e in eyes))
            report['frames_with_lens_pose']+=int(bool(lens_control and lens_control.material) and any(e.lens_pose is not None for e in eyes))
            times.append(time.perf_counter())
            fps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 1000/max(1,timing['total_ms'])
            render_start=time.perf_counter()
            canvas=render(frame,eyes,timing,fps,mirror,overlay,**(lens_control.render_options() if lens_control else {}))
            if lens_control:canvas=lens_control.decorate(canvas)
            timing['render_ms']=(time.perf_counter()-render_start)*1000
            if not args.headless:
                show(title,canvas)
                key=wait_key(title,1)
                if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                    break
                if key==ord('m'): mirror=not mirror
                if key==ord('o'): overlay=not overlay
                if key==ord('s'): save_snapshot(canvas,eyes,frame,lens_control)
                opening_video=key==ord('v') or lens_control.pending=='video'
                if lens_control.handle_pending(key) and opening_video:
                    pipeline.reset_tracking();times.clear()
            if args.image:
                if not args.headless:
                    print('Image processed. Q/Esc closes; U imports lens material.',flush=True)
                    while True:
                        key=wait_key(title,40)
                        if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                            break
                        changed=lens_control.handle_pending(key)
                        if key==ord('m'):mirror=not mirror;changed=True
                        if key==ord('o'):overlay=not overlay;changed=True
                        if key==ord('s'):save_snapshot(canvas,eyes,frame,lens_control)
                        if changed:
                            canvas=lens_control.decorate(render(frame,eyes,timing,0,mirror,overlay,image_mode=True,**lens_control.render_options()))
                            show(title,canvas)
                break
            if args.frames and report['processed_frames']>=args.frames:
                break
        report['status']='ok'
        return 0
    except Exception as exc:
        report['status']='error'
        report['error']=str(exc)
        traceback.print_exc()
        return 1
    finally:
        if camera:
            camera.close()
            if run_started:
                report['camera_received_frames']=camera.sequence
                report['camera_received_fps']=camera.sequence/max(.001,time.perf_counter()-run_started)
        if cap is not None: cap.release()
        if pipeline: pipeline.close()
        cv2.destroyAllWindows()
        if timings:
            # Exclude first-frame warmup where possible. This is not sensor-to-screen latency.
            samples=list(timings)[1:] or list(timings)
            report['timing_window_frames']=len(samples)
            report['processing_ms_median']=float(np.median([t['total_ms'] for t in samples]))
            report['processing_ms_p95']=float(np.percentile([t['total_ms'] for t in samples],95))
            for metric in ['mesh_ms','iris_ms','render_ms']:
                report[metric+'_median']=float(np.median([t.get(metric,0) for t in samples]))
            face_samples=[t['total_ms'] for t in samples if t.get('has_eyes')]
            if face_samples:
                report['with_eyes_processing_ms_median']=float(np.median(face_samples))
                report['with_eyes_processing_ms_p95']=float(np.percentile(face_samples,95))
        if eye_diagnostics:
            report['eye_diagnostics']={
                'samples':len(eye_diagnostics),
                'open_samples':sum(int(e['open']) for e in eye_diagnostics),
                **{f'{k}_median':float(np.median([e[k] for e in eye_diagnostics]))
                   for k in ['opening','source_eye_width','raw_iris_max','inside_iris_max','visible_pixels']},
            }
        if run_started:
            report['elapsed_seconds']=time.perf_counter()-run_started
            report['processed_fps']=report['processed_frames']/report['elapsed_seconds']
        print(json.dumps(report,indent=2),flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    raise SystemExit(main())
