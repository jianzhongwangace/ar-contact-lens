import copy
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from ellseg_adapter import validate_geometry
from eye_pipeline import EyePipeline,ROOT
from lens_renderer import LensTracker,compose_lens,normalize_support,support_at_pose
from test_lens_rendering import make_eye,material

def fake_result(ellipse):
    return {'input_valid':True,'ellipses':{'iris':{'plausible':True,'ellipse':ellipse}}}

class GateTests(unittest.TestCase):
    def test_compatible_ellipse_accepted_and_eyelids_still_clip(self):
        eye=make_eye()
        ellipse,reason=validate_geometry(eye,fake_result([80,80,45,45,0]))
        self.assertEqual(reason,'accepted');self.assertIsNotNone(ellipse)
        eye.geometry_ellipse=ellipse;LensTracker(static=True).update([eye])
        for scale in (1.,1.3,1.4):
            frame=eye.crop.copy();compose_lens(frame,eye,material(),1.,scale)
            self.assertTrue(np.any(frame!=eye.crop))
            np.testing.assert_array_equal(frame[eye.eyelid==0],eye.crop[eye.eyelid==0])
            np.testing.assert_array_equal(frame[eye.pupil>0],eye.crop[eye.pupil>0])

    def test_grossly_incorrect_geometry_rejected(self):
        for ellipse in [[130,80,45,45,0],[80,80,110,100,0],[80,80,5,5,0],[80,80,np.nan,45,0]]:
            with self.subTest(ellipse=ellipse):
                found,_=validate_geometry(make_eye(),fake_result(ellipse));self.assertIsNone(found)

    def test_low_evidence_and_closed_eye_never_use_completed_shape(self):
        for case in ('closed','empty','invalid'):
            eye=make_eye();result=fake_result([80,80,45,45,0])
            if case=='closed':eye.is_open=False
            if case=='empty':eye.iris[:]=0
            if case=='invalid':result['input_valid']=False
            self.assertIsNone(validate_geometry(eye,result)[0])

    def test_rotated_normalization_does_not_rotate_twice(self):
        shape=(180,180);center=np.array([90.,90.]);angle=.35
        mask=np.zeros(shape,np.float32)
        cv2.ellipse(mask,((90.,90.),(100.,60.),np.rad2deg(angle)),1,-1)
        normalized=normalize_support(mask,center,50,30,angle)
        restored=support_at_pose(normalized,shape,np.array([90,90,50,30,angle]),np.array([[1.,0,0],[0,1.,0]]))
        a=mask>.5;b=restored>.5
        self.assertGreater((a&b).sum()/(a|b).sum(),.97)

    def test_unified_center_retained_with_rotated_ellseg_geometry(self):
        eye=make_eye();eye.geometry_ellipse=np.array([80,80,45,36,.2])
        LensTracker(static=True).update([eye])
        self.assertIsNotNone(eye.lens_pose)
        np.testing.assert_allclose(eye.lens_pose[:2],eye.pupil_anchor)
        self.assertAlmostEqual(eye.lens_pose[4],.2)
        eye.is_open=False;LensTracker().update([eye])
        self.assertIsNone(eye.lens_pose)

class ActualModelTests(unittest.TestCase):
    def test_import_namespace_switching_and_fallback(self):
        pipeline=EyePipeline(device='auto',geometry='ellseg',static_image=True)
        try:
            im=cv2.imdecode(np.fromfile(str(ROOT/'test_assets/astronaut.png'),np.uint8),1)
            eyes,t=pipeline.process(im)
            self.assertEqual(len(eyes),2)
            self.assertGreater(t['ellseg_ms'],0)
            for e in eyes:
                self.assertIsNotNone(e.ellseg_result)
                self.assertIn(e.geometry_source,('EllSeg','FaceMesh fallback'))
                self.assertFalse(e.iris[e.eyelid==0].any())
            pipeline.set_geometry('facemesh')
            eyes,t=pipeline.process(im)
            self.assertTrue(all(e.geometry_ellipse is None for e in eyes))
            pipeline.set_geometry('ellseg')
            eyes,_=pipeline.process(np.zeros_like(im));self.assertEqual(eyes,[])
        finally:pipeline.close()

    def test_cache_separates_model_and_material_changes(self):
        from video_app import rendered_cache_path
        from lens_import import LensImportControl
        with tempfile.TemporaryDirectory() as directory:
            file=Path(directory)/'source.mp4';file.write_bytes(b'cache identity test')
            control=LensImportControl(state_file=Path(directory)/'state.json')
            first=rendered_cache_path(file,control,'ellseg')
            self.assertNotEqual(first,rendered_cache_path(file,control,'facemesh'))
            control.material=material()
            second=rendered_cache_path(file,control,'ellseg')
            self.assertNotEqual(first,second)
            control.material.rgba[50,50,0]^=127
            self.assertNotEqual(second,rendered_cache_path(file,control,'ellseg'))

if __name__=='__main__':unittest.main(verbosity=2)
