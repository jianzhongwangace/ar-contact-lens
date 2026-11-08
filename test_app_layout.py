import unittest
from pathlib import Path
import tempfile
import cv2
import numpy as np
from app_layout import WIDTH,HEIGHT,CONTENT_HEIGHT,full_canvas,map_mouse,viewport_size
from eye_pipeline import render,final_eye_view
from lens_import import LensImportControl
from lens_renderer import compose_lens
from test_lens_rendering import make_eye,material

class LayoutTests(unittest.TestCase):
    def test_fullscreen_keys_do_not_exit_application(self):
        from unittest.mock import patch
        from app_layout import wait_key,_fullscreen
        title='fullscreen-key-test'
        _fullscreen.discard(title)
        with patch('app_layout.cv2.setWindowProperty') as set_property:
            with patch('app_layout.cv2.waitKeyEx',return_value=0x7A0000):
                self.assertEqual(wait_key(title),-1)
            self.assertIn(title,_fullscreen)
            with patch('app_layout.cv2.waitKeyEx',return_value=27):
                self.assertEqual(wait_key(title),-1)
                self.assertNotIn(title,_fullscreen)
                self.assertEqual(wait_key(title),27)
            self.assertEqual(set_property.call_count,2)

    def test_highgui_mouse_coordinates_are_not_scaled_twice(self):
        from unittest.mock import patch,Mock
        from app_layout import bind_mouse
        callback=Mock()
        with patch('app_layout.cv2.setMouseCallback') as bind:
            bind_mouse('mouse-test',callback)
        dispatch=bind.call_args.args[1]
        dispatch(cv2.EVENT_LBUTTONUP,1750,900,0,None)
        callback.assert_called_once_with(cv2.EVENT_LBUTTONUP,1750,900,0,None)
        dispatch(cv2.EVENT_LBUTTONUP,-1,900,0,None)
        self.assertEqual(callback.call_count,1)

    def test_aspect_ratio_constant_for_input_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            control=LensImportControl(state_file=Path(directory)/'none.json')
            for h,w in [(1920,1080),(1080,1920),(512,512),(200,1900)]:
                frame=np.zeros((h,w,3),np.uint8)
                base=render(frame,[],{'mesh_ms':0,'iris_ms':0},0)
                final=control.decorate(base)
                self.assertEqual(base.shape,(CONTENT_HEIGHT,WIDTH,3))
                self.assertEqual(final.shape,(HEIGHT,WIDTH,3))
                self.assertEqual(final.shape[1]*9,final.shape[0]*16)

    def test_display_mouse_maps_to_existing_button_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            control=LensImportControl(state_file=Path(directory)/'none.json')
            control.decorate(np.zeros((CONTENT_HEIGHT,WIDTH,3),np.uint8))
            x0,y0,x1,y1=control.rects['video']
            for size in [(1280,720),(1440,810),(960,540)]:
                point=((x0+x1)/2*size[0]/WIDTH,(y0+y1)/2*size[1]/HEIGHT)
                x,y=map_mouse(*point,size)
                control.on_mouse(cv2.EVENT_LBUTTONUP,x,y,0,None)
                self.assertEqual(control.pending,'video')

    def test_closeup_uses_final_pixels_and_respects_mirror(self):
        eye=make_eye();source=eye.crop.copy();compose_lens(source,eye,material())
        actual=final_eye_view(eye,source_result=source)
        expected=cv2.warpAffine(source,eye.affine,(160,160),borderMode=cv2.BORDER_REFLECT_101)
        np.testing.assert_array_equal(actual,expected)
        self.assertTrue(np.any(actual!=eye.crop))
        np.testing.assert_array_equal(final_eye_view(eye,mirror=True,source_result=source),cv2.flip(expected,1))

    def test_original_and_closed_eye_do_not_show_diagnostic_colors(self):
        eye=make_eye()
        np.testing.assert_array_equal(final_eye_view(eye,overlay=False,material=material()),eye.crop)
        eye.is_open=False
        np.testing.assert_array_equal(final_eye_view(eye,material=material()),eye.crop)

    def test_face_loss_clears_closeups(self):
        eye=make_eye();frame=eye.crop.copy();timing={'mesh_ms':0,'iris_ms':0}
        with_eye=render(frame,[eye],timing,0,material=material())
        without=render(frame,[],timing,0,material=material())
        self.assertFalse(np.array_equal(with_eye[98:293,16:348],without[98:293,16:348]))
        self.assertTrue(np.all(without[100:290,20:345]==[33,35,39]))

if __name__=='__main__':unittest.main(verbosity=2)
