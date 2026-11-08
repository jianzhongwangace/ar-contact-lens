import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from lens_material import LensMaterial,prepare_material,ROOT,load_source
from lens_renderer import compose_lens,LensTracker
from eye_pipeline import EyeResult

def make_eye():
    h=w=160
    iris=np.zeros((h,w),np.uint8);cv2.circle(iris,(80,80),45,1,-1)
    pupil=np.zeros_like(iris);cv2.circle(pupil,(80,80),16,1,-1)
    lid=np.zeros_like(iris);cv2.ellipse(lid,(80,80),(65,30),0,0,360,1,-1)
    iris*=lid
    e=EyeResult('right',np.full((h,w,3),70,np.uint8),np.array([[1,0,0],[0,1,0]],np.float64),
        np.array([[15,80],[80,50],[145,80],[80,110]],np.float32),np.array([80,80],np.float32),lid,.3,True,
        iris.astype(np.float32),pupil.astype(np.float32),iris,pupil)
    e.iris_points=np.array([[35,80],[80,35],[125,80],[80,125]],np.float32)
    LensTracker(static=True).update([e]);return e

def material():
    rgba=np.zeros((128,128,4),np.uint8)
    rgba[:,:,:3]=[180,90,45]
    cv2.circle(rgba,(64,64),62,(180,90,45,230),-1)
    cv2.circle(rgba,(64,64),20,(0,0,0,0),-1)
    return LensMaterial(rgba,{})

class RenderingTests(unittest.TestCase):
    def test_closed_eye_and_zero_strength_unchanged(self):
        e=make_eye();frame=e.crop.copy();expected=frame.copy()
        compose_lens(frame,e,material(),0)
        np.testing.assert_array_equal(frame,expected)
        e.is_open=False;compose_lens(frame,e,material(),1)
        np.testing.assert_array_equal(frame,expected)

    def test_pupil_and_eyelid_protection(self):
        e=make_eye();frame=e.crop.copy();expected=frame.copy()
        compose_lens(frame,e,material(),.8)
        self.assertTrue(np.any(frame!=expected))
        np.testing.assert_array_equal(frame[e.eyelid==0],expected[e.eyelid==0])
        np.testing.assert_array_equal(frame[e.pupil>0],expected[e.pupil>0])

    def test_enlargement_still_clipped_by_eyelid(self):
        e=make_eye();frame=e.crop.copy();expected=frame.copy()
        compose_lens(frame,e,material(),.8,1.12)
        np.testing.assert_array_equal(frame[e.eyelid==0],expected[e.eyelid==0])

    def test_reset_on_blink_and_loss(self):
        e=make_eye();tracker=LensTracker()
        tracker.update([e],1);self.assertIsNotNone(e.lens_pose)
        e.is_open=False;tracker.update([e],1.03)
        self.assertIsNone(e.lens_pose);self.assertFalse(tracker.history)
        e.is_open=True;tracker.update([e],1.06)
        self.assertIsNotNone(e.lens_pose)
        tracker.update([],1.09);self.assertFalse(tracker.history)

    def test_nonempty_distant_component_rejected(self):
        e=make_eye();e.iris[:]=0;e.iris[0:10,0:10]=1
        LensTracker().update([e]);self.assertIsNone(e.lens_pose)

    def test_source_geometry_stable_across_crop_change(self):
        e=make_eye();tracker=LensTracker();tracker.update([e],1)
        original=e.lens_pose.copy()
        e.affine[:,2]+=[3,4];e.center+=[3,4];e.iris_points+=[3,4]
        # Move only crop coordinates, not the underlying camera-space eye.
        e.iris=cv2.warpAffine(e.iris,np.array([[1.,0,3],[0,1.,4]]),(160,160))
        tracker.update([e],1.03)
        np.testing.assert_allclose(e.lens_pose[2:],original[2:],atol=1e-3)
        self.assertLess(np.linalg.norm(e.lens_pose[:2]-original[:2]),5.)

class MaterialTests(unittest.TestCase):
    def test_examples_are_transparent_and_preserve_source(self):
        import hashlib
        for path in ROOT.glob('*美瞳*.png'):
            with self.subTest(path=path.name):
                before=hashlib.sha256(path.read_bytes()).hexdigest()
                m=prepare_material(path)
                self.assertEqual(m.rgba.shape,(512,512,4))
                self.assertEqual(m.rgba[256,256,3],0)
                self.assertEqual(m.rgba[0,0,3],0)
                self.assertGreater(int(m.rgba[:,:,3].max()),100)
                self.assertTrue(np.isfinite(m.premultiplied).all())
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
                self.assertTrue((m.directory/'material.json').exists())

    def test_rgba_transparency_and_cancelled_repair(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'花纹.png'
            Image.fromarray(material().rgba).save(path)
            result=prepare_material(path,cache=False)
            self.assertEqual(result.metadata['method'],'preserve-alpha')
            self.assertFalse(result.metadata['calibration']['repair_reflections'])
            self.assertLessEqual(int(result.rgba[:,:,3].max()),230)
            self.assertEqual(result.rgba[256,256,3],0)

    def test_empty_material_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'empty.png'
            Image.new('RGBA',(128,128),(255,255,255,0)).save(path)
            with self.assertRaises(ValueError):prepare_material(path,cache=False)

if __name__=='__main__':unittest.main(verbosity=2)
