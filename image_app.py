"""Local image input with file picker, original/result toggle and explicit saving."""
import argparse
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image, ImageOps
from eye_pipeline import EyePipeline, ROOT, render, blend_eye_region
from lens_import import LensImportControl
from app_layout import open_window,show,bind_mouse,wait_key

def select_image():
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk()
    root.withdraw()
    root.attributes('-topmost',True)
    try:
        filename=filedialog.askopenfilename(title='选择需要检测的图片',parent=root,
            filetypes=[('图片','*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff'),('所有文件','*.*')])
        return Path(filename) if filename else None
    finally:root.destroy()

def read_image(path):
    # Handles Chinese paths and phone EXIF rotation before coordinates are calculated.
    with Image.open(path) as source:
        image=ImageOps.exif_transpose(source).convert('RGB')
        if image.width<32 or image.height<32:
            raise ValueError('Image must be at least 32x32 pixels')
        return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)

def save_result(frame,eyes,canvas,lens_control=None):
    folder=ROOT/'output'/('image-'+time.strftime('%Y%m%d-%H%M%S')+'-'+str(time.time_ns()%1_000_000))
    folder.mkdir(parents=True)
    result=frame.copy()
    for eye in eyes:
        if lens_control and lens_control.material:
            from lens_renderer import compose_lens
            compose_lens(result,eye,lens_control.material,lens_control.strength,lens_control.lens_scale)
        else:blend_eye_region(result,eye)
    for filename,image in [('result.png',result),('diagnostic.png',canvas)]:
        cv2.imencode('.png',image)[1].tofile(str(folder/filename))
    for eye in eyes:
        for name,mask in [('iris',eye.iris),('pupil',eye.pupil),('eyelid',eye.eyelid)]:
            cv2.imencode('.png',mask*255)[1].tofile(str(folder/f'{eye.side}_{name}.png'))
    print('Saved locally:',folder,flush=True)
    return folder

def main():
    parser=argparse.ArgumentParser(description='FaceMesh + LightIrisNet for local images')
    parser.add_argument('--image',type=Path,help='Omit to open file picker')
    parser.add_argument('--lens',type=Path,help='Lens product photo or RGBA texture')
    parser.add_argument('--device',choices=['auto','cpu','cuda','dml','onnx-cpu'],default='auto')
    parser.add_argument('--geometry',choices=['ellseg','facemesh'],default='ellseg')
    parser.add_argument('--input-width',type=int,default=256)
    parser.add_argument('--input-height',type=int,default=160)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--save',action='store_true',help='Explicitly save annotated output and masks locally')
    args=parser.parse_args()
    if args.headless and not args.image:parser.error('--headless needs --image PATH')
    path=args.image or select_image()
    if path is None:return 0
    pipeline=None
    try:
        frame=read_image(path)
        print('Loading models...',flush=True)
        pipeline=EyePipeline(device=args.device,input_width=args.input_width,
                             input_height=args.input_height,static_image=True,geometry=args.geometry)
        eyes,timing=pipeline.process(frame)
        overlay,mirror=True,False
        lens_control=LensImportControl()
        lens_control.geometry_mode=pipeline.geometry_mode
        def switch_geometry(mode):
            nonlocal eyes,timing
            pipeline.set_geometry(mode)
            eyes,timing=pipeline.process(frame)
        lens_control.geometry_callback=switch_geometry
        if args.lens:lens_control.load(args.lens,persist=not args.headless)
        def current_canvas():
            view=render(frame,eyes,timing,0,mirror,overlay,image_mode=True,**lens_control.render_options())
            return lens_control.decorate(view) if lens_control else view
        canvas=current_canvas()
        print(f'Image processed: {len(eyes)} eyes, {timing["total_ms"]:.1f} ms, device={pipeline.predictor.device}',flush=True)
        if args.save:save_result(frame,eyes,canvas,lens_control)
        if args.headless:return 0
        title='AR eyes | Local image'
        open_window(title)
        bind_mouse(title,lens_control.on_mouse)
        show(title,canvas)
        while True:
            key=wait_key(title,40)
            if key in (27,ord('q')) or cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:break
            changed=False
            if lens_control.handle_pending(key):changed=True
            if key==ord('o'):overlay=not overlay;changed=True
            if key==ord('m'):mirror=not mirror;changed=True
            if key==ord('s'):save_result(frame,eyes,canvas,lens_control)
            if key==ord('l'):
                new_path=select_image()
                if new_path:
                    try:
                        new_frame=read_image(new_path)
                        new_eyes,new_timing=pipeline.process(new_frame)
                    except Exception as exc:
                        # Keep the previous image/result visible if the new file fails.
                        import tkinter as tk
                        from tkinter import messagebox
                        root=tk.Tk();root.withdraw()
                        try:messagebox.showerror('图片无法处理',str(exc),parent=root)
                        finally:root.destroy()
                    else:
                        path,frame,eyes,timing=new_path,new_frame,new_eyes,new_timing
                        overlay=True;changed=True
            if changed:
                canvas=current_canvas()
                show(title,canvas)
        return 0
    finally:
        if pipeline:pipeline.close()
        cv2.destroyAllWindows()

if __name__=='__main__':raise SystemExit(main())
