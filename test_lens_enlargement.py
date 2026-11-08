"""Enlargement must actually cover sclera while retaining eyelid/pupil limits."""
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from PIL import Image
from lens_import import LensImportControl
from lens_renderer import compose_lens
from lens_settings import clamp_lens_scale
from test_lens_rendering import make_eye,material


class EnlargementTests(unittest.TestCase):
    def test_enlargement_reaches_visible_sclera(self):
        eye=make_eye();natural=eye.crop.copy();large=eye.crop.copy()
        compose_lens(natural,eye,material(),.8,1.)
        compose_lens(large,eye,material(),.8,1.25)
        sclera=(eye.iris==0)&(eye.eyelid>0)
        old_count=np.sum(np.any(natural!=eye.crop,axis=2)&sclera)
        new_count=np.sum(np.any(large!=eye.crop,axis=2)&sclera)
        self.assertGreater(new_count,old_count+100)

    def test_largest_diameter_cannot_paint_skin_or_pupil(self):
        eye=make_eye();frame=eye.crop.copy()
        compose_lens(frame,eye,material(),1.,1.4)
        np.testing.assert_array_equal(frame[eye.eyelid==0],eye.crop[eye.eyelid==0])
        np.testing.assert_array_equal(frame[eye.pupil>0],eye.crop[eye.pupil>0])

    def test_aperture_does_not_expand_with_outer_ring(self):
        eye=make_eye();tex=material();tex.metadata={'calibration':{'hole':20/64}}
        frame=eye.crop.copy();compose_lens(frame,eye,tex,1.,1.4)
        # Just outside the real pupil: a uniformly scaled transparent aperture
        # would erase the print here. Annular mapping keeps this region colored.
        self.assertTrue(np.any(frame[80,98]!=eye.crop[80,98]))
        np.testing.assert_array_equal(frame[80,80],eye.crop[80,80])

    def test_closed_or_untracked_eye_stays_original_at_large_size(self):
        for closed in (True,False):
            eye=make_eye()
            if closed:eye.is_open=False
            else:eye.lens_pose=None
            frame=eye.crop.copy();compose_lens(frame,eye,material(),1.,1.4)
            np.testing.assert_array_equal(frame,eye.crop)

    def test_edge_of_frame_and_invalid_scale_are_bounded(self):
        eye=make_eye();frame=eye.crop[40:140,60:140].copy()
        compose_lens(frame,eye,material(),1.,999,np.array([[1,0,-60],[0,1,-40],[0,0,1]],float))
        self.assertEqual(frame.shape,(100,80,3))
        self.assertEqual(clamp_lens_scale(999),1.4)
        self.assertEqual(clamp_lens_scale(float('nan')),1.)
        self.assertEqual(clamp_lens_scale(-1),.9)

    def test_slider_and_presets_update_and_remember_extended_range(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);image_path=folder/'花纹.png'
            Image.fromarray(material().rgba).save(image_path)
            control=LensImportControl(folder/'state.json');control.load(image_path)
            control.decorate(np.zeros((752,1920,3),np.uint8))
            x0,y0,x1,y1=control.rects['scale_slider'];y=(y0+y1)//2
            control.on_mouse(cv2.EVENT_LBUTTONDOWN,x0,y,cv2.EVENT_FLAG_LBUTTON,None)
            control.handle_pending()
            control.on_mouse(cv2.EVENT_MOUSEMOVE,x1,y,cv2.EVENT_FLAG_LBUTTON,None)
            control.handle_pending()
            control.on_mouse(cv2.EVENT_LBUTTONUP,x1,y,0,None);control.handle_pending()
            self.assertEqual(control.lens_scale,1.4)
            restored=LensImportControl(folder/'state.json')
            self.assertEqual(restored.lens_scale,1.4)
            control.pending='natural_size';control.handle_pending();self.assertEqual(control.lens_scale,1.)
            control.pending='enlarged_size';control.handle_pending();self.assertEqual(control.lens_scale,1.3)

if __name__=='__main__':unittest.main(verbosity=2)
