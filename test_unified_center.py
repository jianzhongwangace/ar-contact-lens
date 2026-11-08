"""Rendering must not reintroduce eccentric UV warping when pupil/iris differ."""
import unittest
import cv2
import numpy as np
from center_tracking import concentric_uv
from lens_renderer import LensTracker,compose_lens
from test_pupil_alignment import displaced_eye
from test_lens_rendering import material

class UnifiedCenterTests(unittest.TestCase):
    def test_rendered_ring_is_symmetric_around_pupil_at_all_sizes(self):
        for offset in ([8,0],[-8,0],[0,6]):
            for scale in (1.,1.2,1.4):
                with self.subTest(offset=offset,scale=scale):
                    eye=displaced_eye(offset)
                    # Remove physical occlusion deliberately: this checks the
                    # generated ring, independently from legitimate eyelid cuts.
                    eye.eyelid[:]=1
                    LensTracker(static=True).update([eye],0)
                    eye.lens_support[:]=1
                    tex=material();tex.metadata={'calibration':{'hole':.32}}
                    image=eye.crop.copy();compose_lens(image,eye,tex,.8,scale)
                    mask=np.any(image!=eye.crop,axis=2)
                    cx,cy=np.rint(eye.pupil_anchor).astype(int)
                    indices=np.flatnonzero(mask[cy])
                    self.assertGreater(len(indices),0)
                    self.assertLessEqual(abs((cx-indices[0])-(indices[-1]-cx)),2)
                    left=int(mask[cy,:cx].sum());right=int(mask[cy,cx+1:].sum())
                    self.assertLessEqual(abs(left-right),2)

    def test_renderer_uses_one_center_even_if_diagnostic_anchor_is_changed(self):
        eye=displaced_eye([8,0]);LensTracker(static=True).update([eye],0)
        first=eye.crop.copy();compose_lens(first,eye,material(),.8,1.2)
        eye.pupil_anchor+=np.array([10.,-5.])
        second=eye.crop.copy();compose_lens(second,eye,material(),.8,1.2)
        np.testing.assert_array_equal(first,second)

    def test_unified_center_during_movement_fallback_and_recovery(self):
        tracker=LensTracker()
        for i in range(70):
            eye=displaced_eye([min(9.,i*.2),0])
            if 35<=i<39:eye.pupil_prob[:]=0
            tracker.update([eye],i/30)
            self.assertIsNotNone(eye.lens_pose)
            np.testing.assert_allclose(eye.lens_pose[:2],eye.pupil_anchor,atol=1e-10)
            self.assertLess(np.max(np.abs(eye.lens_pose[2:4]-45)),.1)

    def test_rotation_and_preview_transform_do_not_split_centers(self):
        eye=displaced_eye([8,0]);LensTracker(static=True).update([eye],0)
        eye.lens_pose[4]=.25
        frame=eye.crop.copy()
        transform=np.array([[.8,0,3],[0,.8,4],[0,0,1.]])
        compose_lens(frame,eye,material(),.8,1.2,transform)
        self.assertTrue(np.isfinite(frame).all())
        np.testing.assert_allclose(eye.lens_pose[:2],eye.pupil_anchor)

    def test_texture_mapping_has_no_direction_dependent_stretch(self):
        theta=np.linspace(0,2*np.pi,360,endpoint=False)
        for scale in (.9,1.,1.2,1.4):
            for radius in (.1,.3,.6,.95):
                coords=np.stack([np.cos(theta),np.sin(theta)],axis=1)*radius
                uv=concentric_uv(coords,.41,scale)
                lengths=np.linalg.norm(uv,axis=1)
                self.assertLess(np.ptp(lengths),1e-6)

if __name__=='__main__':unittest.main(verbosity=2)
