"""Standalone EllSeg viewer. FaceMesh optionally crops eyes; EllSeg does all eye inference."""
import argparse
import json
from pathlib import Path
import time
import traceback
import cv2
import numpy as np
from PIL import Image,ImageOps
from ellseg_core import ROOT,EllSegPredictor,annotate,transform_ellipse

def choose_source():
    import tkinter as tk
    from tkinter import filedialog
    choice={}
    root=tk.Tk();root.title('EllSeg 独立测试');root.geometry('470x280');root.resizable(False,False)
    tk.Label(root,text='EllSeg 虹膜 / 瞳孔分割与椭圆',font=('Microsoft YaHei',16)).pack(pady=20)
    mode=tk.StringVar(value='face')
    tk.Radiobutton(root,text='人像 / 摄像头：自动定位双眼',variable=mode,value='face').pack(anchor='w',padx=45)
    tk.Radiobutton(root,text='单眼近景：直接运行 EllSeg',variable=mode,value='eye').pack(anchor='w',padx=45)
    def select(kind):
        if kind=='camera':path=0
        else:
            patterns='*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff' if kind=='image' else '*.mp4 *.mov *.avi *.mkv *.webm'
            path=filedialog.askopenfilename(parent=root,title='选择'+('图片' if kind=='image' else '视频'),filetypes=[('媒体',patterns),('所有文件','*.*')])
            if not path:return
        choice.update(kind=kind,path=path,mode=mode.get());root.destroy()
    row=tk.Frame(root);row.pack(pady=22)
    for text,kind in [('图片','image'),('离线视频','video'),('USB 相机','camera')]:
        tk.Button(row,text=text,width=12,command=lambda k=kind:select(k)).pack(side='left',padx=5)
    tk.Label(root,text='灰度眼部模型 · 默认本机推理 · 此窗口不应用美瞳花纹',fg='#555').pack()
    root.mainloop();return choice or None

class EyeCropper:
    def __init__(self,static=False):
        from mediapipe_compat import load_mediapipe
        mp=load_mediapipe()
        self.mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=static,max_num_faces=1,
                                                refine_landmarks=True,min_detection_confidence=.5,min_tracking_confidence=.5)
    def close(self):self.mesh.close()
    def extract(self,frame):
        result=self.mesh.process(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:return []
        h,w=frame.shape[:2]
        points=np.array([[p.x*w,p.y*h] for p in result.multi_face_landmarks[0].landmark],np.float32)
        crops=[]
        for side,indices in [('right',(33,133)),('left',(362,263))]:
            a,b=sorted(points[list(indices)],key=lambda p:p[0])
            width=float(np.linalg.norm(b-a))
            if width<12:continue
            center=(a+b)/2;ux=(b-a)/width;uy=np.array([-ux[1],ux[0]])
            cw=width*1.7;ch=cw*.75
            origin=center-ux*cw/2-uy*ch/2
            source=np.array([origin,origin+ux*cw,origin+uy*ch],np.float32)
            target=np.array([[0,0],[319,0],[0,239]],np.float32)
            affine=cv2.getAffineTransform(source,target)
            crop=cv2.warpAffine(frame,affine,(320,240),borderMode=cv2.BORDER_REFLECT_101)
            crops.append((side,crop,affine))
        return crops

def process_frame(frame,predictor,cropper=None):
    start=time.perf_counter()
    crops=cropper.extract(frame) if cropper else [('eye',frame,np.array([[1.,0,0],[0,1.,0]]))]
    records=[];view=frame.copy()
    for side,crop,affine in crops:
        result=predictor.predict(crop)
        inverse=cv2.invertAffineTransform(affine)
        source_ellipses={}
        for name,record in result['ellipses'].items():
            values=record['ellipse']
            ellipse=transform_ellipse(values,inverse) if values is not None else None
            source_ellipses[name]={**record,'ellipse':None if ellipse is None else ellipse.tolist()}
            if record['plausible'] and ellipse is not None:
                x,y,a,b,angle=ellipse
                color=(80,255,80) if name=='iris' else (0,190,255)
                cv2.ellipse(view,((x,y),(2*a,2*b),np.rad2deg(angle)),color,2,cv2.LINE_AA)
                cv2.drawMarker(view,(round(x),round(y)),color,cv2.MARKER_CROSS,9,1)
        records.append({'side':side,'crop':crop,'affine':affine,'result':result,'source_ellipses':source_ellipses})
    return view,records,(time.perf_counter()-start)*1000

def paste_fit(canvas,image,x,y,w,h):
    scale=min(w/image.shape[1],h/image.shape[0]);dw=max(1,round(image.shape[1]*scale));dh=max(1,round(image.shape[0]*scale))
    canvas[y+(h-dh)//2:y+(h-dh)//2+dh,x+(w-dw)//2:x+(w-dw)//2+dw]=cv2.resize(image,(dw,dh))

def make_view(frame,records,elapsed,device,mode):
    canvas=np.full((820,1440,3),24,np.uint8)
    paste_fit(canvas,frame,10,45,760,710)
    cv2.putText(canvas,'EllSeg | iris (green) / pupil (orange)',(16,29),cv2.FONT_HERSHEY_SIMPLEX,.65,(235,235,235),1,cv2.LINE_AA)
    for i,record in enumerate(records[:2]):
        y=40+i*330
        result=record['result'];crop=record['crop']
        labels=np.zeros_like(crop);labels[result['mask']==1]=(80,180,80);labels[result['mask']==2]=(0,170,255)
        cv2.putText(canvas,f'{record["side"]} eye - {result["processing_ms"]:.0f} ms',(795,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1)
        paste_fit(canvas,crop,790,y+15,300,225)
        paste_fit(canvas,annotate(crop,result),1100,y+15,320,240)
        paste_fit(canvas,labels,795,y+248,105,70)
        status=' / '.join(f'{name}: '+('plausible' if r['plausible'] else 'uncertain') for name,r in result['ellipses'].items())
        cv2.putText(canvas,status,(912,y+272),cv2.FONT_HERSHEY_SIMPLEX,.38,(215,215,215),1)
        cv2.putText(canvas,'Full ellipse is estimated, not an eyelid mask.',(912,y+295),cv2.FONT_HERSHEY_SIMPLEX,.37,(180,190,205),1)
    if not records:cv2.putText(canvas,'No face detected - face the camera',(800,120),cv2.FONT_HERSHEY_SIMPLEX,.65,(160,200,255),1)
    cv2.putText(canvas,f'{device} | mode={mode} | total {elapsed:.0f} ms | Q/Esc exit | S save frame | Space pause',
                (16,790),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,230,230),1,cv2.LINE_AA)
    return canvas

def json_records(records):
    return [{'side':r['side'],'crop_shape':list(r['crop'].shape[:2]),'source_to_crop':r['affine'].tolist(),
             'ellipses_source':r['source_ellipses'],'input_valid':r['result']['input_valid'],
             'processing_ms':r['result']['processing_ms']} for r in records]

def save_snapshot(output,frame,view,records):
    output.mkdir(parents=True,exist_ok=True)
    cv2.imencode('.png',frame)[1].tofile(str(output/'result.png'))
    cv2.imencode('.png',view)[1].tofile(str(output/'preview.png'))
    for record in records:
        side=record['side']
        cv2.imencode('.png',record['result']['mask'])[1].tofile(str(output/f'{side}-labels.png'))
        cv2.imencode('.png',annotate(record['crop'],record['result']))[1].tofile(str(output/f'{side}-eye.png'))
    (output/'ellipses.json').write_text(json.dumps(json_records(records),ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description='Standalone EllSeg: images, videos, USB camera')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--image',type=Path);group.add_argument('--video',type=Path);group.add_argument('--camera',type=int)
    parser.add_argument('--mode',choices=['face','eye'],default='face')
    parser.add_argument('--device',choices=['auto','cpu','cuda','dml','onnx-cpu'],default='auto')
    parser.add_argument('--headless',action='store_true');parser.add_argument('--frames',type=int,default=0)
    parser.add_argument('--save',action='store_true',help='Explicitly save results (video has no audio)')
    parser.add_argument('--report',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.frames<0:parser.error('--frames must be non-negative')
    if args.image:kind,path='image',args.image
    elif args.video:kind,path='video',args.video
    elif args.camera is not None:kind,path='camera',args.camera
    else:
        if args.headless:parser.error('Specify --image, --video or --camera')
        selected=choose_source()
        if not selected:return 0
        kind,path,args.mode=selected['kind'],selected['path'],selected['mode']
    if args.headless and kind=='camera' and args.frames<=0:parser.error('Headless camera requires --frames N')
    output=args.output or ROOT/'outputs'/time.strftime('%Y%m%d-%H%M%S')
    cap=None;cropper=None;writer=None;json_file=None
    report={'status':'starting','frames':0,'eye_predictions':0,'mode':args.mode,'kind':kind}
    from collections import deque
    timings=deque(maxlen=1000);source_fps=30.;paused=False;expected_frames=0
    try:
        print('Loading standalone EllSeg...',flush=True)
        predictor=EllSegPredictor(args.device)
        report['device']=predictor.device;report['adapter']=predictor.adapter
        if args.mode=='face':cropper=EyeCropper(static=kind=='image')
        if kind=='image':
            with Image.open(path) as source:frame=cv2.cvtColor(np.asarray(ImageOps.exif_transpose(source).convert('RGB')),cv2.COLOR_RGB2BGR)
        else:
            cap=cv2.VideoCapture(path if kind=='camera' else str(path),cv2.CAP_DSHOW if kind=='camera' else cv2.CAP_FFMPEG)
            if not cap.isOpened():raise RuntimeError('Cannot open camera/video. Close other camera apps or check the file.')
            if kind=='camera':
                cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
                cap.set(cv2.CAP_PROP_FPS,30);cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            else:cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,1)
            fps=cap.get(cv2.CAP_PROP_FPS);source_fps=fps if np.isfinite(fps) and 0<fps<=240 else 30.
            count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
            expected_frames=int(count) if np.isfinite(count) and count>0 else 0
        if not args.headless:
            cv2.namedWindow('EllSeg standalone',cv2.WINDOW_NORMAL|cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow('EllSeg standalone',1200,683)
        while True:
            if not paused:
                if kind!='image':
                    ok,frame=cap.read()
                    if not ok:
                        if kind=='camera' or report['frames']==0:raise RuntimeError('No readable camera/video frame')
                        if expected_frames and report['frames']<expected_frames-2:
                            raise RuntimeError(f'Video decode failed at frame {report["frames"]}; expected {expected_frames}')
                        break
                report['frame_size']=[frame.shape[1],frame.shape[0]]
                annotated,records,elapsed=process_frame(frame,predictor,cropper)
                view=make_view(annotated,records,elapsed,predictor.device,args.mode)
                report['frames']+=1;report['eye_predictions']+=len(records);timings.append(elapsed)
                if args.save:
                    if kind=='image':save_snapshot(output,annotated,view,records)
                    else:
                        if writer is None:
                            output.mkdir(parents=True,exist_ok=True)
                            writer=cv2.VideoWriter(str(output/'ellseg-result.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),source_fps,(frame.shape[1],frame.shape[0]))
                            if not writer.isOpened():raise RuntimeError('Cannot create result video')
                            json_file=(output/'ellipses.jsonl').open('w',encoding='utf-8')
                        writer.write(annotated)
                        json_file.write(json.dumps({'frame':report['frames']-1,'eyes':json_records(records)},ensure_ascii=False)+'\n')
                if not args.headless:cv2.imshow('EllSeg standalone',view)
                if kind=='image':paused=True
            if args.headless:
                if kind=='image' or (args.frames and report['frames']>=args.frames):break
                continue
            key=cv2.waitKey(20 if paused else 1)&255
            if key in (27,ord('q')) or cv2.getWindowProperty('EllSeg standalone',cv2.WND_PROP_VISIBLE)<1:break
            if key==ord(' ') and kind!='image':paused=not paused
            if key==ord('s'):save_snapshot(output/f'snapshot-{report["frames"]}',annotated,view,records)
            if args.frames and report['frames']>=args.frames:break
        report['status']='ok'
        if args.save:report['output']=str(output.resolve())
        return 0
    except Exception as exc:
        report.update(status='error',error=str(exc));traceback.print_exc();return 1
    finally:
        if cap:cap.release()
        if cropper:cropper.close()
        if writer:writer.release()
        if json_file:json_file.close()
        cv2.destroyAllWindows()
        if timings:report['median_frame_processing_ms']=float(np.median(timings))
        if args.report:
            args.report.parent.mkdir(parents=True,exist_ok=True)
            args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':raise SystemExit(main())
