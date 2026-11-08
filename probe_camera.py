"""Measure camera delivery independently of AI; stores numbers only."""
import json,time
from pathlib import Path
import cv2
import numpy as np
from camera_app import open_camera

def main():
    results=[]
    for width,height in [(1920,1080),(1280,720),(640,480)]:
        cap,frame=open_camera(0,'dshow',width,height)
        if cap is None:
            results.append({'requested':[width,height],'error':'unavailable'})
            continue
        try:
            for _ in range(3):cap.read()
            start=time.perf_counter();times=[];brightness=[]
            while time.perf_counter()-start<6:
                ok,frame=cap.read()
                if not ok:break
                times.append(time.perf_counter())
                brightness.append(float(frame[::8,::8].mean()))
            elapsed=time.perf_counter()-start
            fourcc=int(cap.get(cv2.CAP_PROP_FOURCC))
            result={'requested':[width,height],'actual':[int(cap.get(3)),int(cap.get(4))],
                    'fourcc':''.join(chr((fourcc>>(8*i))&255) for i in range(4)),
                    'reported_fps':cap.get(cv2.CAP_PROP_FPS),'measured_fps':len(times)/elapsed,
                    'exposure':cap.get(cv2.CAP_PROP_EXPOSURE),'auto_exposure':cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                    'mean_brightness':float(np.mean(brightness)) if brightness else None}
            print(json.dumps(result),flush=True);results.append(result)
        finally:cap.release()
    Path('logs/camera-delivery.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
