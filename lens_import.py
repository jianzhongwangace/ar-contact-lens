"""Shared local lens-image import button for OpenCV camera/photo windows."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from lens_material import prepare_material
from lens_settings import MIN_LENS_SCALE,MAX_LENS_SCALE,ENLARGED_LENS_SCALE,clamp_lens_scale

ROOT=Path(__file__).resolve().parent
STATE_FILE=ROOT/'lens-selection.json'


def read_lens_image(path):
    path=Path(path)
    if path.stat().st_size>25*1024*1024:
        raise ValueError('图片文件过大，请选择 25 MB 以内的素材。')
    with Image.open(path) as source:
        if source.width*source.height>20_000_000:
            raise ValueError('图片尺寸过大，请先缩小到 2000 万像素以内。')
        image=ImageOps.exif_transpose(source).convert('RGBA')
        if min(image.size)<8:
            raise ValueError('图片太小，请选择至少 8×8 像素的素材。')
        if image.getchannel('A').getextrema()[1]==0:
            raise ValueError('这张图片完全透明，请选择带有可见花纹的素材。')
        return image.copy()


def choose_lens_file():
    import tkinter as tk
    from tkinter import filedialog
    root=tk.Tk();root.withdraw();root.attributes('-topmost',True)
    try:
        return filedialog.askopenfilename(parent=root,title='导入美瞳花纹图片',
            filetypes=[('美瞳图片','*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff'),('所有文件','*.*')])
    finally:root.destroy()


def show_import_error(error):
    import tkinter as tk
    from tkinter import messagebox
    root=tk.Tk();root.withdraw();root.attributes('-topmost',True)
    try:messagebox.showerror('无法导入美瞳图片',str(error),parent=root)
    finally:root.destroy()


class LensImportControl:
    """Callbacks only queue actions; dialogs run on the application main loop."""
    HEIGHT=204

    def __init__(self,state_file=STATE_FILE):
        self.state_file=Path(state_file)
        self.path=None
        self.image=None
        self.material=None
        self.strength=.70
        self.lens_scale=ENLARGED_LENS_SCALE
        self.pending=None
        self.video_callback=None
        self.geometry_mode='ellseg'
        self.geometry_callback=None
        self.alignment_debug=False
        self.dragging_scale=False
        self.rects={}
        self.panel=None
        self.panel_width=0
        self.status='未选择素材 · 推荐使用透明背景的正面美瞳花纹 PNG'
        self.fonts={}
        for size in (22,17,15):
            self.fonts[size]=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size)
        if self.state_file.exists():
            try:
                state=json.loads(self.state_file.read_text(encoding='utf-8'))
                saved=state.get('path')
                if saved and not Path(saved).exists() and (ROOT/Path(saved).name).exists():
                    saved=str(ROOT/Path(saved).name)
                if saved:self.load(saved,persist=False,calibration=state.get('calibration'))
                self.strength=float(np.clip(state.get('strength',.70),0,1))
                self.lens_scale=clamp_lens_scale(state.get('scale',ENLARGED_LENS_SCALE))
            except (OSError,ValueError,Image.DecompressionBombError):
                self.status='上次素材无法读取，请重新导入美瞳图片。'

    def load(self,path,persist=True,calibration=None):
        path=Path(path).resolve()
        read_lens_image(path)
        material=prepare_material(path,calibration)
        image=material.thumbnail
        # Persist before replacing the current selection so failures leave it intact.
        if persist:
            self.state_file.parent.mkdir(parents=True,exist_ok=True)
            self.state_file.write_text(json.dumps({'path':str(path),'calibration':material.metadata['calibration'],
                'strength':self.strength,'scale':self.lens_scale},ensure_ascii=False),encoding='utf-8')
        self.path,self.image,self.material=path,image,material
        self.status='已应用试戴 · 商品图提取为近似效果，可点击校准' if material.metadata['method']!='preserve-alpha' else '透明花纹已应用试戴'
        self.panel=None

    def persist(self):
        if self.material:
            self.state_file.write_text(json.dumps({'path':str(self.path),'calibration':self.material.metadata['calibration'],
                'strength':self.strength,'scale':self.lens_scale},ensure_ascii=False),encoding='utf-8')

    def render_options(self):
        return {'material':self.material,'strength':self.strength,'lens_scale':self.lens_scale,
                'alignment_debug':self.alignment_debug}

    def clear(self):
        if self.state_file.exists():self.state_file.unlink()
        self.path,self.image=None,None
        self.material=None
        self.status='未选择素材 · 推荐使用透明背景的正面美瞳花纹 PNG'
        self.panel=None

    def on_mouse(self,event,x,y,flags,param):
        scale_rect=self.rects.get('scale_slider')
        if scale_rect:
            x0,y0,x1,y1=scale_rect
            if event==cv2.EVENT_LBUTTONDOWN and x0<=x<x1 and y0<=y<y1:
                self.dragging_scale=True
            if self.dragging_scale and event in (cv2.EVENT_LBUTTONDOWN,cv2.EVENT_MOUSEMOVE,cv2.EVENT_LBUTTONUP):
                ratio=float(np.clip((x-x0)/max(1,x1-x0),0,1))
                value=round(MIN_LENS_SCALE+ratio*(MAX_LENS_SCALE-MIN_LENS_SCALE),2)
                commit=event==cv2.EVENT_LBUTTONUP or (event==cv2.EVENT_MOUSEMOVE and not flags&cv2.EVENT_FLAG_LBUTTON)
                self.pending=('set_scale',value,commit)
                if commit:self.dragging_scale=False
                return
        if event!=cv2.EVENT_LBUTTONUP:return
        for action,(x0,y0,x1,y1) in self.rects.items():
            if action=='scale_slider':continue
            if x0<=x<x1 and y0<=y<y1:
                self.pending=action
                break

    def handle_pending(self,key=None):
        if key==ord('u'):self.pending='import'
        if key==ord('c'):self.pending='calibrate'
        if key in (ord('+'),ord('=')):self.pending='stronger'
        if key==ord('-'):self.pending='weaker'
        if key==ord('v'):self.pending='video'
        if key==ord('d'):self.pending='alignment'
        if key==ord('g'):self.pending='geometry'
        action,self.pending=self.pending,None
        if action is None:return False
        try:
            if action=='geometry':
                target='facemesh' if self.geometry_mode=='ellseg' else 'ellseg'
                if self.geometry_callback:self.geometry_callback(target)
                self.geometry_mode=target;self.panel=None;return True
            if action=='alignment':
                self.alignment_debug=not self.alignment_debug
                self.panel=None;return True
            if isinstance(action,tuple) and action[0]=='set_scale':
                self.lens_scale=clamp_lens_scale(action[1])
                if action[2]:self.persist()
                self.panel=None;return True
            if action in ('natural_size','enlarged_size'):
                self.lens_scale=1. if action=='natural_size' else ENLARGED_LENS_SCALE
                self.persist();self.panel=None;return True
            if action=='video':
                from video_app import choose_video,play_video
                path=choose_video()
                if not path:return False
                if self.video_callback:self.video_callback(path)
                else:play_video(path,lens_control=self)
                return True
            if action in ('stronger','weaker','bigger','smaller'):
                if action=='stronger':self.strength=min(1.,self.strength+.05)
                if action=='weaker':self.strength=max(0.,self.strength-.05)
                if action=='bigger':self.lens_scale=clamp_lens_scale(round(self.lens_scale+.02,2))
                if action=='smaller':self.lens_scale=clamp_lens_scale(round(self.lens_scale-.02,2))
                self.persist();self.panel=None;return True
            if action=='calibrate':
                if not self.material:return False
                from lens_calibration import calibrate_material
                adjusted=calibrate_material(self.path,self.material)
                if adjusted is None:return False
                self.material=adjusted;self.image=adjusted.thumbnail
                self.persist();self.panel=None;return True
            if action=='clear':
                self.clear();return True
            path=choose_lens_file()
            if not path:return False
            self.load(path)
            return True
        except Exception as exc:
            show_import_error(exc)
            return False

    def _fit_text(self,text,font,width):
        if font.getlength(text)<=width:return text
        while text and font.getlength(text+'…')>width:text=text[:-1]
        return text+'…'

    def _build_panel(self,width):
        panel=Image.new('RGB',(width,self.HEIGHT),(35,39,46))
        draw=ImageDraw.Draw(panel)
        draw.line((0,0,width,0),fill=(68,74,83))
        draw.rounded_rectangle((20,20,250,83),radius=10,fill=(69,104,189))
        draw.text((40,36),'导入美瞳图片',font=self.fonts[22],fill='white')
        # Checkerboard makes transparent PNG regions visible without changing alpha.
        for y in range(14,94,10):
            for x in range(275,355,10):
                shade=180 if ((x-275)//10+(y-14)//10)%2 else 220
                draw.rectangle((x,y,x+9,y+9),fill=(shade,shade,shade))
        if self.image:
            thumb=self.image.copy();thumb.thumbnail((80,80),Image.Resampling.LANCZOS)
            panel.paste(thumb,(275+(80-thumb.width)//2,14+(80-thumb.height)//2),thumb)
        name=self.path.name if self.path else '美瞳素材'
        draw.text((375,22),self._fit_text(name,self.fonts[17],width-555),font=self.fonts[17],fill=(245,245,245))
        draw.text((375,57),self._fit_text(self.status,self.fonts[15],width-555),font=self.fonts[15],fill=(200,208,220))
        draw.rounded_rectangle((width-150,24,width-24,78),radius=8,fill=(62,69,79))
        draw.text((width-111,38),'移除',font=self.fonts[17],fill=(235,235,235))
        for text,x0,x1 in [('校准素材',20,170),('强度 −',190,310),('强度 +',320,440),
                            ('直径 −',650,770),('直径 +',780,900)]:
            draw.rounded_rectangle((x0,105,x1,140),radius=6,fill=(62,69,79))
            draw.text((x0+15,111),text,font=self.fonts[15],fill=(235,235,235))
        draw.text((465,111),f'强度 {self.strength:.0%}',font=self.fonts[15],fill='white')
        draw.text((920,111),f'着色直径 {self.lens_scale:.0%}（相对原虹膜）',font=self.fonts[15],fill='white')
        for text,x0,x1 in [('自然 100%',1320,1480),('放大 130%',1500,1670)]:
            draw.rounded_rectangle((x0,105,x1,140),radius=6,fill=(62,69,79))
            draw.text((x0+15,111),text,font=self.fonts[15],fill='white')
        draw.text((1700,111),'O 对比 / S 保存',font=self.fonts[15],fill='white')
        draw.rounded_rectangle((20,156,250,194),radius=7,fill=(69,104,189))
        draw.text((39,163),'导入离线视频',font=self.fonts[17],fill='white')
        draw.text((275,164),'支持 1080×1920 竖屏 · MP4 / MOV · 逐帧试戴预览（V）',font=self.fonts[17],fill=(220,225,235))
        draw.text((1070,162),'90%',font=self.fonts[15],fill=(210,215,225))
        draw.text((1825,162),'140%',font=self.fonts[15],fill=(210,215,225))
        draw.text((920,55),'D 瞳孔对齐诊断：'+('开启' if self.alignment_debug else '关闭'),font=self.fonts[15],fill=(200,210,225))
        draw.rounded_rectangle((1270,48,1680,87),radius=6,fill=(62,69,79))
        draw.text((1285,57),'G 几何模型：'+('EllSeg + 可见区域' if self.geometry_mode=='ellseg' else '原方案 FaceMesh'),font=self.fonts[15],fill='white')
        draw.rounded_rectangle((1130,170,1800,177),radius=3,fill=(84,92,105))
        thumb_x=1130+(1800-1130)*(self.lens_scale-MIN_LENS_SCALE)/(MAX_LENS_SCALE-MIN_LENS_SCALE)
        draw.ellipse((thumb_x-9,165,thumb_x+9,183),fill=(124,184,248))
        self.panel=cv2.cvtColor(np.asarray(panel),cv2.COLOR_RGB2BGR)
        self.panel_width=width

    def decorate(self,canvas):
        from app_layout import full_canvas,CONTENT_HEIGHT,WIDTH
        h,w=CONTENT_HEIGHT,WIDTH
        if self.panel is None or self.panel_width!=w:self._build_panel(w)
        self.rects={'import':(20,h+20,251,h+84),'clear':(w-150,h+24,w-23,h+79)}
        self.rects['geometry']=(1270,h+48,1681,h+88)
        for name,x0,x1 in [('calibrate',20,170),('weaker',190,310),('stronger',320,440),('smaller',650,770),('bigger',780,900)]:
            self.rects[name]=(x0,h+105,x1,h+141)
        self.rects['video']=(20,h+156,251,h+195)
        self.rects['natural_size']=(1320,h+105,1481,h+141)
        self.rects['enlarged_size']=(1500,h+105,1671,h+141)
        self.rects['scale_slider']=(1130,h+153,1800,h+197)
        output=full_canvas(canvas).copy()
        output[h:h+self.HEIGHT]=self.panel
        return output
