"""Resizable/fullscreen windows displaying a fixed 16:9 logical canvas."""
import ctypes
import os
import cv2
import numpy as np

WIDTH,HEIGHT=1920,1080
CONTENT_HEIGHT=752
MATERIAL_HEIGHT=204
TRANSPORT_HEIGHT=124
_windows={}
_fullscreen=set()

def viewport_size():
    sw,sh=1920,1080
    if os.name=='nt':
        sw=ctypes.windll.user32.GetSystemMetrics(0)
        sh=ctypes.windll.user32.GetSystemMetrics(1)
    # Initial window size only; maximized/fullscreen windows may grow freely.
    units=max(20,int(min(1440/16,(sw-64)/16,(sh-120)/9)))
    return units*16,units*9

def open_window(title):
    _windows[title]=viewport_size()
    _fullscreen.discard(title)
    cv2.namedWindow(title,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(title,*_windows[title])

def full_canvas(canvas):
    """Reserve transport area even in photo/camera mode, preventing ratio jumps."""
    if canvas.shape[1]!=WIDTH or canvas.shape[0]>HEIGHT:
        raise ValueError('App canvas must be 1920 pixels wide and at most 1080 high')
    if canvas.shape[0]==HEIGHT:return canvas
    output=np.full((HEIGHT,WIDTH,3),24,np.uint8)
    output[:canvas.shape[0]]=canvas
    return output

def show(title,canvas):
    canvas=full_canvas(canvas)
    if title not in _windows:open_window(title)
    # Submit the logical image unchanged. HighGUI scales it to the live client
    # area, including when paused or maximized, and maps mouse coordinates back
    # into this image. Pre-resizing here causes fixed-size content and double
    # scaling of clicks.
    cv2.imshow(title,canvas)

def map_mouse(x,y,size):
    return int(x*WIDTH/size[0]),int(y*HEIGHT/size[1])

def bind_mouse(title,callback):
    def dispatch(event,x,y,flags,param):
        if 0<=x<WIDTH and 0<=y<HEIGHT:
            callback(event,x,y,flags,param)
    cv2.setMouseCallback(title,dispatch)

def toggle_fullscreen(title):
    enabled=title not in _fullscreen
    cv2.setWindowProperty(title,cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL)
    if enabled:_fullscreen.add(title)
    else:_fullscreen.discard(title)

def wait_key(title,delay=1):
    """F11 toggles fullscreen; Escape first exits fullscreen, then exits app."""
    key=cv2.waitKeyEx(delay)
    if key in (0x7A0000,0xFFC8,16777274): # Windows, GTK, Qt F11
        toggle_fullscreen(title)
        return -1
    if key==27 and title in _fullscreen:
        toggle_fullscreen(title)
        return -1
    return key&255 if key>=0 else -1
