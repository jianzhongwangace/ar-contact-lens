"""Regression checks for the meaningful boundaries of the two-model integration."""
import unittest
import cv2
import numpy as np
import torch
from eye_pipeline import EyePipeline, EyeResult, component_near, visible_masks, render, ROOT, blend_eye_region


class MaskTests(unittest.TestCase):
    def test_closed_eye_cannot_leak_effect(self):
        ip=np.ones((32,32),np.float32)
        iris,pupil=visible_masks(ip,ip,np.ones_like(ip,np.uint8),np.array([16,16]),False)
        self.assertFalse(iris.any())
        self.assertFalse(pupil.any())

    def test_pixels_outside_eyelid_never_render(self):
        ip=np.ones((32,32),np.float32)
        lid=np.zeros((32,32),np.uint8)
        lid[8:24,4:28]=1
        iris,pupil=visible_masks(ip,ip,lid,np.array([16,16]),True)
        self.assertTrue(iris.any())
        self.assertFalse(iris[lid==0].any())
        self.assertFalse(pupil[lid==0].any())

    def test_center_component_beats_larger_false_positive(self):
        mask=np.zeros((32,32),np.uint8)
        mask[2:12,2:12]=1
        mask[19:23,19:23]=1
        got=component_near(mask,np.array([20,20]))
        self.assertEqual(int(got.sum()),16)
        self.assertEqual(int(got[5,5]),0)

    def test_pupil_cannot_escape_iris(self):
        iris=np.zeros((32,32),np.float32)
        iris[10:22,10:22]=1
        a,b=visible_masks(iris,np.ones_like(iris),np.ones_like(iris,np.uint8),np.array([16,16]),True)
        self.assertFalse(b[a==0].any())


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline=EyePipeline(device='cpu',input_width=192,input_height=128,static_image=True)

    @classmethod
    def tearDownClass(cls):
        cls.pipeline.close()

    def test_supplied_checkpoint_and_empty_frame(self):
        self.assertGreater(self.pipeline.predictor.parameters,4_000_000)
        frame=np.zeros((480,640,3),np.uint8)
        eyes,timing=self.pipeline.process(frame)
        self.assertEqual(eyes,[])
        self.assertEqual(timing['iris_ms'],0)
        self.assertEqual(render(frame,eyes,timing,0).shape[2],3)

    def test_inference_path_matches_full_upstream_forward(self):
        predictor=self.pipeline.predictor
        torch.manual_seed(42)
        x=torch.randn(1,3,128,192,device=predictor.device)
        with torch.inference_mode():
            full=predictor.model(x)
            iris,pupil=predictor.segmentation_logits(x)
        torch.testing.assert_close(iris,full['iris'],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(pupil,full['pupil'],rtol=1e-5,atol=1e-6)

    def test_real_photo_two_eye_pipeline(self):
        photo=ROOT/'test_assets/astronaut.png'
        if not photo.exists():
            self.skipTest('Optional public-domain skimage astronaut fixture not installed')
        frame=cv2.imdecode(np.fromfile(str(photo),np.uint8),1)
        eyes,t=self.pipeline.process(frame)
        self.assertEqual(len(eyes),2)
        for eye in eyes:
            self.assertTrue(np.isfinite(eye.iris_prob).all())
            self.assertTrue(np.isfinite(eye.pupil_prob).all())
            self.assertGreater(int(eye.iris.sum()),0)
            self.assertFalse(eye.iris[eye.eyelid==0].any())
            self.assertFalse(eye.pupil[eye.iris==0].any())
            # Crop coordinates round-trip back into the original image.
            inverse=cv2.invertAffineTransform(eye.affine)
            recovered=cv2.transform(cv2.transform(eye.contour[None],inverse),eye.affine)[0]
            np.testing.assert_allclose(recovered,eye.contour,atol=1e-3)
        self.assertTrue(render(frame,eyes,t,1).size)
        # A lost face does not retain the previous frame's masks.
        empty,_=self.pipeline.process(np.zeros_like(frame))
        self.assertEqual(empty,[])


class OptimizedRenderingTests(unittest.TestCase):
    def test_roi_warp_matches_full_frame_at_edges_and_rotations(self):
        for angle,center in [(0,(130,100)),(25,(130,100)),(-35,(0,0)),(40,(315,235)),(0,(-100,-100))]:
            with self.subTest(angle=angle,center=center):
                frame=np.full((240,320,3),110,np.uint8)
                iris=np.zeros((64,96),np.uint8)
                pupil=np.zeros_like(iris)
                cv2.ellipse(iris,(48,32),(22,15),0,0,360,1,-1)
                cv2.circle(pupil,(48,32),7,1,-1)
                inverse=cv2.getRotationMatrix2D((48,32),angle,1.25)
                inverse[:,2]+=np.array(center)-[48,32]
                affine=cv2.invertAffineTransform(inverse)
                contour=np.array([[5,32],[48,10],[90,32],[48,54]],np.float32)
                eye=EyeResult('right',np.zeros((64,96,3),np.uint8),affine,contour,np.array([48,32]),iris,.3,True,
                              iris.astype(float),pupil.astype(float),iris,pupil)
                expected=frame.copy()
                im=cv2.warpAffine(iris,inverse,(320,240),flags=cv2.INTER_NEAREST)>0
                pm=cv2.warpAffine(pupil,inverse,(320,240),flags=cv2.INTER_NEAREST)>0
                ring=im&~pm
                expected[ring]=(.5*expected[ring]+.5*np.array([80,220,70])).astype(np.uint8)
                expected[pm]=(.4*expected[pm]+.6*np.array([0,170,255])).astype(np.uint8)
                cv2.polylines(expected,[cv2.transform(contour[None],inverse)[0].astype(np.int32)],True,(255,220,0),1,cv2.LINE_AA)
                actual=frame.copy()
                blend_eye_region(actual,eye)
                # OpenCV's fixed-point affine rounding may differ at exact half-pixels.
                self.assertLess(float(np.mean(np.any(actual!=expected,axis=2))),.0005)

    def test_portrait_image_fits_canvas(self):
        frame=np.zeros((2000,500,3),np.uint8)
        view=render(frame,[],{'mesh_ms':0,'iris_ms':0},0,image_mode=True)
        self.assertEqual(view.shape,(752,1920,3))

    def test_panorama_image_fits_canvas(self):
        frame=np.zeros((300,1920,3),np.uint8)
        view=render(frame,[],{'mesh_ms':0,'iris_ms':0},0,image_mode=True)
        self.assertEqual(view.shape,(752,1920,3))


class ImageInputTests(unittest.TestCase):
    def test_unicode_path_and_exif_rotation(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        from image_app import read_image
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'测试照片.jpg'
            photo=Image.new('RGB',(80,40),(255,0,0))
            exif=photo.getexif();exif[274]=6
            photo.save(path,exif=exif)
            decoded=read_image(path)
            self.assertEqual(decoded.shape,(80,40,3))
            self.assertGreater(decoded[20,20,2],240)


class CameraNegotiationTests(unittest.TestCase):
    def test_resolution_change_does_not_leave_uncompressed_five_fps_mode(self):
        from unittest.mock import patch
        from camera_app import open_camera
        class Driver:
            def __init__(self):self.format=cv2.VideoWriter_fourcc(*'YUY2')
            def isOpened(self):return True
            def set(self,key,value):
                if key in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FPS):
                    self.format=cv2.VideoWriter_fourcc(*'YUY2')
                elif key==cv2.CAP_PROP_FOURCC:self.format=value
                return True
            def read(self):return True,np.zeros((480,640,3),np.uint8)
            def release(self):pass
        driver=Driver()
        with patch('camera_app.cv2.VideoCapture',return_value=driver):
            cap,frame=open_camera(0,'dshow',1920,1080)
        self.assertIs(cap,driver)
        self.assertEqual(driver.format,cv2.VideoWriter_fourcc(*'MJPG'))


if __name__=='__main__':
    unittest.main(verbosity=2)
