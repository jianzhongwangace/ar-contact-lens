"""Optional visual correction of source geometry; applied once per material."""
from PIL import Image,ImageDraw,ImageTk
from lens_material import load_source,prepare_material

def calibrate_material(path,material):
    import tkinter as tk
    from tkinter import ttk,messagebox
    source=load_source(path)
    h,w=source.shape[:2]
    root=tk.Tk();root.title('校准美瞳素材');root.attributes('-topmost',True)
    state={'result':None,'job':None,'images':[]}
    c=dict(material.metadata['calibration'])
    images=ttk.Frame(root);images.pack(padx=12,pady=10)
    labels=[ttk.Label(images) for _ in range(3)]
    for i,label in enumerate(labels):label.grid(row=0,column=i,padx=4)
    controls=ttk.Frame(root);controls.pack(fill='x',padx=18)
    values={}
    for row,(key,label,low,high) in enumerate([
        ('cx','中心 X',0,w),('cy','中心 Y',0,h),('rx','外圈横向半径',8,w/2),
        ('ry','外圈纵向半径',8,h/2),('angle','方向（度）',0,180),('hole','中央透明区比例',.03,.7)]):
        values[key]=tk.DoubleVar(value=c[key])
        ttk.Label(controls,text=label,width=20).grid(row=row,column=0,sticky='w')
        scale=tk.Scale(controls,variable=values[key],from_=low,to=high,orient='horizontal',
                       resolution=.01 if key=='hole' else .5,length=520,command=lambda _:schedule())
        scale.grid(row=row,column=1,sticky='ew')
    repair=tk.BooleanVar(value=c['repair_reflections'])
    ttk.Checkbutton(root,text='近似修补反光（相邻/对侧纹理，不代表恢复真实图案）',variable=repair,command=lambda:schedule()).pack(pady=5)
    message=tk.StringVar(value='左：原图内外边界　中：深瞳背景　右：浅瞳背景')
    ttk.Label(root,textvariable=message).pack(pady=5)
    def config():
        return {**c,**{key:value.get() for key,value in values.items()},'repair_reflections':repair.get()}
    def refresh():
        state['job']=None
        try:
            settings=config()
            prepared=prepare_material(path,settings,size=256,cache=False)
            original=Image.fromarray(source).convert('RGB')
            # Draw oriented ellipse outlines on scaled source image.
            import cv2,numpy as np
            pixels=np.asarray(original).copy()
            for ratio,color in [(1,(0,220,80)),(settings['hole'],(255,150,0))]:
                cv2.ellipse(pixels,((settings['cx'],settings['cy']),
                    (2*settings['rx']*ratio,2*settings['ry']*ratio),settings['angle']),color,4)
            original=Image.fromarray(pixels);original.thumbnail((270,270))
            previews=[original]
            for bg in [(55,34,22),(145,113,74)]:
                preview=Image.new('RGB',(256,256),bg)
                preview.paste(prepared.thumbnail,(0,0),prepared.thumbnail);previews.append(preview)
            state['images']=[ImageTk.PhotoImage(im,master=root) for im in previews]
            for label,im in zip(labels,state['images']):label.configure(image=im)
        except Exception as exc:message.set(str(exc))
    def schedule():
        if state['job']:root.after_cancel(state['job'])
        state['job']=root.after(250,refresh)
    def accept():
        try:state['result']=prepare_material(path,config())
        except Exception as exc:messagebox.showerror('校准失败',str(exc),parent=root);return
        root.destroy()
    buttons=ttk.Frame(root);buttons.pack(pady=12)
    ttk.Button(buttons,text='应用校准',command=accept).pack(side='left',padx=12)
    ttk.Button(buttons,text='取消',command=root.destroy).pack(side='left',padx=12)
    refresh();root.mainloop()
    return state['result']
