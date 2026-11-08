"""Local product-photo to lens material processing. Originals are never changed.

White-background extraction and radial reflection repair are approximations,
not recovery of a manufacturer's print or measured optical transmittance.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT=Path(__file__).resolve().parent
VERSION=7
# Reviewed reflection regions for the user's three source files. SHA256, not
# filename, prevents applying these annotations to an unrelated product.
# Each region is normalized cx, cy, radius_x, radius_y, rotation_deg.
EXAMPLE_PROFILES={
 '13f4ccc502a81930d74ae8de46e8eb871e1e6291aff8a1e83e3b1a26d5ed1477':{
   'reflection_regions':[[.39,.385,.039,.016,-43],[.407,.407,.022,.011,-43]]},
 '3dbd0288b08a01386f4fd8a0020da4a35b2c68933377f554bd1832722ce505b4':{
   'normalized_geometry':[.495,.490,.305,.315,0],
   'hole':.41,'reflection_regions':[[.358,.356,.075,.078,-20],[.293,.256,.105,.022,-40],
                                    [.751,.705,.13,.024,-45]]},
 'abadf0d4f07a3b5f801056ba38af4dcae8ed4a07149248f0e04694f2ced8e96e':{
   'normalized_geometry':[.502,.491,.260,.275,0],
   'hole':.40,'flatten_lighting':.85,'radial_repair':True,'reflection_regions':[[.386,.36,.082,.103,-25],[.33,.285,.12,.035,-42],
                                    [.699,.661,.13,.054,-53]]},
}

def srgb_to_linear(x):
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)

def linear_to_srgb(x):
    x=np.clip(x,0,1)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)

def smoothstep(a,b,x):
    t=np.clip((x-a)/(b-a),0,1)
    return t*t*(3-2*t)

def load_source(path):
    p=Path(path)
    if p.stat().st_size>25*1024*1024:raise ValueError('素材超过 25 MB，请先缩小图片。')
    with Image.open(p) as im:
        if im.width*im.height>20_000_000:raise ValueError('素材超过 2000 万像素。')
        im=ImageOps.exif_transpose(im).convert('RGBA')
        if min(im.size)<8:raise ValueError('素材尺寸过小。')
        if im.getchannel('A').getextrema()[1]==0:raise ValueError('素材完全透明。')
        return np.asarray(im).copy()

@dataclass
class LensMaterial:
    rgba: np.ndarray
    metadata: dict
    directory: Path | None = None

    def __post_init__(self):
        self.alpha=self.rgba[:,:,3].astype(np.float32)/255
        self.color=srgb_to_linear(self.rgba[:,:,:3].astype(np.float32)/255)
        self.premultiplied=np.dstack([self.color*self.alpha[:,:,None],self.alpha]).astype(np.float32)

    @property
    def thumbnail(self):return Image.fromarray(self.rgba,'RGBA')

def detect_calibration(rgba):
    h,w=rgba.shape[:2]
    ratio=min(1.,800/max(h,w))
    small=cv2.resize(rgba,None,fx=ratio,fy=ratio,interpolation=cv2.INTER_AREA)
    transparent=bool(np.min(rgba[:,:,3])<250)
    if transparent:
        mask=(small[:,:,3]>24).astype(np.uint8)*255
    else:
        rgb=small[:,:,:3].astype(np.float32)/255
        gray=cv2.cvtColor(small[:,:,:3],cv2.COLOR_RGB2GRAY)
        # Restrict to dark/colored printing; shadows alone should not dominate.
        saturation=rgb.max(2)-rgb.min(2)
        mask=((gray<165)|((gray<205)&(saturation>.13))).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    candidates=[c for c in contours if len(c)>=5 and cv2.contourArea(c)>.005*mask.size]
    if not candidates:raise ValueError('未找到环形花纹。请使用正面镜片图或透明花纹 PNG。')
    c=max(candidates,key=cv2.contourArea)
    (cx,cy),(dw,dh),angle=cv2.fitEllipse(c)
    if min(dw,dh)<15 or max(dw,dh)/min(dw,dh)>2.2:
        raise ValueError('花纹过小或倾斜过大，请选择更清晰的正面图。')
    trim=1. if transparent else .96
    return {'cx':float(cx/ratio),'cy':float(cy/ratio),'rx':float(dw/ratio/2*trim),
            'ry':float(dh/ratio/2*trim),'angle':float(angle),'hole':None,
            'repair_reflections':not transparent,'transparent_source':transparent}

def sample_polar(rgba,calibration,radial=256,angular=720):
    c=calibration
    radius=np.linspace(0,1,radial,dtype=np.float32)[:,None]
    theta=np.linspace(0,2*np.pi,angular,endpoint=False,dtype=np.float32)[None,:]
    angle=np.deg2rad(c['angle'])
    xx=c['rx']*radius*np.cos(theta);yy=c['ry']*radius*np.sin(theta)
    mx=(c['cx']+np.cos(angle)*xx-np.sin(angle)*yy).astype(np.float32)
    my=(c['cy']+np.sin(angle)*xx+np.cos(angle)*yy).astype(np.float32)
    return cv2.remap(rgba,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(255,255,255,0))

def prepare_material(path,calibration=None,size=512,cache=True):
    path=Path(path).resolve()
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    rgba=load_source(path)
    c=dict(calibration or detect_calibration(rgba))
    if calibration is None and digest in EXAMPLE_PROFILES:
        c.update(EXAMPLE_PROFILES[digest])
        geometry=c.pop('normalized_geometry',None)
        if geometry:
            hh,ww=rgba.shape[:2]
            c.update(cx=geometry[0]*ww,cy=geometry[1]*hh,rx=geometry[2]*ww,ry=geometry[3]*hh,angle=geometry[4])
    key=hashlib.sha256((digest+json.dumps(c,sort_keys=True)+str(size)+str(VERSION)).encode()).hexdigest()[:20]
    directory=ROOT/'materials'/key
    if cache and (directory/'material.json').exists():
        meta=json.loads((directory/'material.json').read_text(encoding='utf-8'))
        with Image.open(directory/'lens_rgba.png') as im:out=np.asarray(im.convert('RGBA')).copy()
        return LensMaterial(out,meta,directory)
    polar=sample_polar(rgba,c).astype(np.float32)/255
    rgb=polar[:,:,:3].copy()
    transparent=c['transparent_source']
    # Inner boundary is measured before reflection repair.
    radial_dark=np.quantile(1-rgb.mean(2),.7,axis=1)
    if c.get('hole') is None:
        if transparent:
            active=np.flatnonzero(np.quantile(polar[:,:,3],.7,axis=1)>.08)
        else:
            baseline=float(np.median(radial_dark[:20]))
            threshold=baseline+.32*(float(np.quantile(radial_dark,.85))-baseline)
            active=np.flatnonzero((np.arange(len(radial_dark))>len(radial_dark)*.12)&(radial_dark>threshold))
        c['hole']=float(np.clip((active[0]/255-.035) if len(active) else .35,.05 if transparent else .32,.62))
    repaired=np.zeros(polar.shape[:2],np.uint8)
    if c.get('repair_reflections') and not transparent:
        luminance=rgb.mean(2)
        median=np.median(luminance,axis=1)[:,None]
        # A reflection is brighter than other angular samples at a similar radius.
        chroma=rgb.max(2)-rgb.min(2)
        candidate=((luminance>.60)&(luminance-median>.22)&(chroma<.13)).astype(np.uint8)
        if c.get('reflection_regions'):
            source_mask=np.zeros_like(rgba)
            hh,ww=rgba.shape[:2]
            for cx,cy,rx,ry,rotation in c['reflection_regions']:
                cv2.ellipse(source_mask,((cx*ww,cy*hh),(rx*ww*2,ry*hh*2),rotation),(255,255,255,255),-1)
            candidate=(sample_polar(source_mask,c)[:,:,0]>40).astype(np.uint8)
        candidate[:int(c['hole']*255)]=0
        count,labels,stats,_=cv2.connectedComponentsWithStats(candidate)
        for label in range(1,count):
            if stats[label,cv2.CC_STAT_AREA]>=30:repaired[labels==label]=1
        repaired=cv2.dilate(repaired,np.ones((5,5),np.uint8))
        if c.get('reflection_regions'):
            # Replace complete affected angular sectors for these reviewed
            # photos. This avoids leaving an oval-shaped specular patch seam.
            columns=np.sum(repaired,axis=0)>3
            repaired[int(c['hole']*255):,columns]=1
        # Copy an unreflected radial sector, preserving texture rather than
        # smearing it into concentric lines. This remains approximate repair.
        donor=rgb.copy()
        unresolved=repaired>0
        for shift in (240,480,180,540,360,90,630):
            available=np.roll(repaired,shift,axis=1)==0
            use=unresolved&available
            candidate_rgb=np.roll(rgb,shift,axis=1)
            donor[use]=candidate_rgb[use]
            unresolved[use]=False
        if c.get('radial_repair'):
            # For the dark granular third print, circular interpolation gives
            # smoother junctions than copying petals/sectors. Restore a bounded
            # amount of donor microtexture without copying its shading.
            donor_detail=donor-cv2.GaussianBlur(donor,(0,0),2)
            indices=np.arange(rgb.shape[1])
            for row in range(rgb.shape[0]):
                valid=repaired[row]==0
                if valid.sum()>30:
                    for channel in range(3):
                        baseline=np.interp(indices,indices[valid],rgb[row,valid,channel],period=rgb.shape[1])
                        donor[row,:,channel]=np.clip(baseline+.35*donor_detail[row,:,channel],0,1)
        blend=cv2.GaussianBlur(repaired.astype(np.float32),(0,0),6,2)[:,:,None]
        rgb=rgb*(1-blend)+donor*blend
    if c.get('flatten_lighting',0) and not transparent:
        # The third reviewed source contains strong baked-in directional shading.
        # Normalize only its low-frequency angular lighting, retain radial print.
        extended=np.concatenate([rgb[:,-80:],rgb,rgb[:,:80]],axis=1)
        low=cv2.GaussianBlur(extended,(0,0),12,4)[:,80:-80]
        target=np.median(low,axis=1)[:,None,:]
        gain=np.clip((target+.025)/(low+.025),.45,2.2)
        amount=float(c['flatten_lighting'])
        rgb=np.clip(rgb*((1-amount)+amount*gain),0,1)
    r=np.linspace(0,1,rgb.shape[0])[:,None]
    ring=smoothstep(c['hole'],c['hole']+.055,r)*(1-smoothstep(.965,1.,r))
    if transparent:
        alpha=polar[:,:,3]*ring
        color=rgb
    else:
        # Approximate removal of a bright background. At least 0.55 opacity in
        # strong printing prevents excessively dark color recovery from division.
        ink=1-rgb.min(2)
        density=smoothstep(.035,.40,ink)
        opacity=np.maximum(.55,1-rgb.min(2))
        alpha=density*opacity*ring
        # RGB white-background unmixing is a heuristic, not measured transmittance.
        unmixed=np.clip(1-(1-rgb)/np.maximum(opacity[:,:,None],.1),0,1)
        color=.35*unmixed+.65*rgb
    packed=np.dstack([color*alpha[:,:,None],alpha]).astype(np.float32)
    grid=(np.arange(size,dtype=np.float32)+.5)/size*2-1
    gx,gy=np.meshgrid(grid,grid)
    rr=np.sqrt(gx*gx+gy*gy)
    theta=np.mod(np.arctan2(gy,gx),2*np.pi)
    # Add one angular column for interpolation across the seam.
    packed=np.concatenate([packed,packed[:,:1]],axis=1)
    sampled=cv2.remap(packed,(theta/(2*np.pi)*720).astype(np.float32),
                      (rr*255).astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    sampled[rr>=1]=0
    a=sampled[:,:,3]
    out=np.dstack([np.where(a[:,:,None]>.001,sampled[:,:,:3]/np.maximum(a[:,:,None],.001),0),a])
    out=np.clip(np.rint(out*255),0,255).astype(np.uint8)
    meta={'version':VERSION,'source':str(path),'source_sha256':digest,'calibration':c,
          'method':'preserve-alpha' if transparent else 'bright-background approximation',
          'reflection_repair_fraction':float(repaired.mean()),
          'warning':'Reflection repair and alpha estimation are approximate; verify against source/product.',
          'size':size}
    material=LensMaterial(out,meta,directory if cache else None)
    if cache:
        directory.mkdir(parents=True,exist_ok=True)
        Image.fromarray(out,'RGBA').save(directory/'lens_rgba.png')
        (directory/'material.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
        marked=rgba[:,:,:3].copy()
        ellipse=((c['cx'],c['cy']),(c['rx']*2,c['ry']*2),c['angle'])
        cv2.ellipse(marked,ellipse,(0,200,70),3)
        inner=((c['cx'],c['cy']),(c['rx']*2*c['hole'],c['ry']*2*c['hole']),c['angle'])
        cv2.ellipse(marked,inner,(255,140,0),3)
        Image.fromarray(marked).save(directory/'calibration.png')
        for name,bg in [('dark',(55,34,22)),('light',(145,113,74))]:
            background=Image.new('RGB',(size,size),bg)
            tex=Image.fromarray(out,'RGBA');background.paste(tex,(0,0),tex)
            background.save(directory/f'preview_{name}.png')
    return material

def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('files',nargs='*',type=Path)
    p.add_argument('--examples',action='store_true')
    args=p.parse_args()
    files=list(ROOT.glob('*美瞳*.png')) if args.examples else args.files
    for path in files:
        material=prepare_material(path)
        print(path.name,material.metadata['calibration'],material.directory,flush=True)

if __name__=='__main__':main()
