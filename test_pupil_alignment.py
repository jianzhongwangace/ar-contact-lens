"""Inner-hole geometry, current-time tracking and pupil fit acceptance."""
import unittest
import numpy as np
import cv2
from center_tracking import track_center,concentric_uv,pose_basis
from lens_renderer import LensTracker,estimate_pupil,compose_lens
from test_lens_rendering import make_eye,material


def displaced_eye(offset):
    eye=make_eye()
    eye.pupil[:]=0
    center=tuple(np.rint(eye.center+offset).astype(int))
    cv2.circle(eye.pupil,center,12,1,-1)
    eye.pupil_prob=eye.pupil.astype(np.float32)
    return eye


class CenterTests(unittest.TestCase):
    def test_constant_velocity_low_lag_at_different_rates(self):
        for fps in (15,30,60):
            state=None;errors=[]
            for i in range(3*fps):
                position=np.array([100+10*i/fps,100.])
                state=track_center(position,state,1/fps,20)
                if i>fps:errors.append(abs(state['position'][0]-position[0]))
            self.assertLess(np.median(errors),10/fps)

    def test_small_persistent_motion_eventually_converges(self):
        state=track_center([100,100],None,1/30,20)
        for _ in range(60):state=track_center([100.4,100],state,1/30,20)
        self.assertLess(abs(state['position'][0]-100.4),.01)

    def test_moving_then_stopping_converges(self):
        state=None
        for i in range(60):state=track_center([100+i/3,100],state,1/30,20)
        for _ in range(45):state=track_center([100+59/3,100],state,1/30,20)
        self.assertLess(abs(state['position'][0]-(100+59/3)),.02)

    def test_stationary_noise_is_reduced(self):
        rng=np.random.default_rng(42);state=None;raw=[];filtered=[]
        for i in range(240):
            measurement=np.array([100.,100.])+rng.normal(0,.65,2)
            state=track_center(measurement,state,1/30,20)
            if i>30:raw.append(measurement);filtered.append(state['position'])
        self.assertTrue(np.all(np.std(filtered,axis=0)<np.std(raw,axis=0)*.5))

    def test_outlier_rejected_confirmed_saccade_follows(self):
        state=track_center([100,100],None,1/30,20)
        outlier=track_center([108,100],state,1/30,20)
        np.testing.assert_array_equal(outlier['position'],[100,100])
        confirmed=track_center([108,100],outlier,1/30,20)
        np.testing.assert_array_equal(confirmed['position'],[108,100])
        resumed=track_center([120,90],confirmed,.5,20)
        np.testing.assert_array_equal(resumed['position'],[120,90])

    def test_rapid_continuous_motion_does_not_stay_locked_as_outlier(self):
        state=track_center([100,100],None,1/30,20)
        for i in range(1,8):
            current=np.array([100+8*i,100])
            state=track_center(current,state,1/30,20)
            if i>=2:self.assertLess(np.linalg.norm(state['position']-current),1.)

    def test_iris_pupil_agreement_follows_first_moving_frame(self):
        state=track_center([100,100],None,1/30,20)
        state=track_center([108,100],state,1/30,20,motion_hint=[7.5,.1])
        self.assertLess(np.linalg.norm(state['position']-[108,100]),.5)


class AlignmentTests(unittest.TestCase):
    def test_offcenter_pupil_moves_both_inner_and_outer_center(self):
        eye=displaced_eye([8,2]);LensTracker(static=True).update([eye],0)
        np.testing.assert_allclose(eye.lens_pose[:2],eye.pupil_anchor,atol=1e-5)
        np.testing.assert_allclose(eye.lens_pose[2:4],[45,45],atol=1e-5)
        np.testing.assert_allclose(eye.pupil_anchor,[88,82],atol=.5)
        yy,xx=np.indices(eye.pupil_protection.shape)
        weights=eye.pupil_protection
        center=np.array([(xx*weights).sum(),(yy*weights).sum()])/weights.sum()
        np.testing.assert_allclose(center,eye.pupil_anchor,atol=.5)

    def test_pupil_motion_with_stationary_iris_is_followed(self):
        tracker=LensTracker()
        for i in range(100):
            offset=min(8.,i*.15)
            eye=displaced_eye([offset,0]);tracker.update([eye],i/30)
        self.assertGreater(eye.pupil_anchor[0],87.5)
        np.testing.assert_allclose(eye.lens_pose[:2],eye.pupil_anchor,atol=.01)

    def test_specular_hole_does_not_shift_ellipse_center(self):
        eye=displaced_eye([8,0])
        eye.pupil_prob[72:79,88:96]=0
        result=estimate_pupil(eye,eye.center,45,45)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(eye.center+result[:2]*45,[88,80],atol=.6)

    def test_invalid_pupil_never_becomes_a_valid_anchor(self):
        eye=make_eye();eye.pupil_prob=np.ones_like(eye.pupil_prob)
        LensTracker(static=True).update([eye],0)
        self.assertIsNone(eye.pupil_observed)
        self.assertEqual(eye.pupil_confidence,0)
        np.testing.assert_allclose(eye.pupil_anchor,eye.lens_pose[:2])

    def test_same_source_geometry_after_crop_change(self):
        tracker=LensTracker();eye=displaced_eye([8,0]);tracker.update([eye],0)
        anchor=eye.pupil_anchor.copy()
        shift=np.array([[1.,0,3],[0,1.,4]])
        eye.affine[:,2]+=[3,4];eye.center+=[3,4];eye.iris_points+=[3,4]
        for attr in ('iris','iris_prob','pupil','pupil_prob','eyelid'):
            setattr(eye,attr,cv2.warpAffine(getattr(eye,attr),shift,(160,160)))
        tracker.update([eye],1/30)
        np.testing.assert_allclose(eye.pupil_anchor,anchor,atol=.1)

    def test_blink_clears_anchor(self):
        eye=displaced_eye([8,0]);tracker=LensTracker();tracker.update([eye],0)
        eye.is_open=False;tracker.update([eye],1/30)
        self.assertIsNone(eye.pupil_anchor);self.assertIsNone(eye.pupil_observed)

    def test_unrelated_static_photo_does_not_borrow_pupil(self):
        tracker=LensTracker(static=True)
        first=displaced_eye([8,0]);tracker.update([first],0)
        second=make_eye();second.pupil_prob[:]=0
        tracker.update([second],1/30)
        self.assertEqual(second.pupil_source,'iris fallback')
        self.assertFalse(second.pupil_protection.any())


class UVTests(unittest.TestCase):
    def test_aperture_center_and_outer_boundary(self):
        hole=.35;scale=1.2
        theta=np.linspace(0,2*np.pi,96,endpoint=False)
        directions=np.column_stack([np.cos(theta),np.sin(theta)])
        inner=directions*hole/scale
        uv=concentric_uv(inner,hole,scale)
        np.testing.assert_allclose(np.linalg.norm(uv,axis=1),hole,atol=1e-6)
        np.testing.assert_allclose(np.linalg.norm(concentric_uv(directions,hole,scale),axis=1),1,atol=1e-6)
        np.testing.assert_allclose(concentric_uv(np.zeros((1,2)),hole,scale),0,atol=1e-6)

    def test_radial_mapping_is_monotone_and_same_in_opposite_directions(self):
        hole=.4
        for direction in [[1,0],[0,1],[-1,0],[0,-1]]:
            coords=np.linspace(0,1,100)[:,None]*direction
            uv=concentric_uv(coords,hole,1.2)
            self.assertTrue(np.all(np.diff(np.linalg.norm(uv,axis=1))>0))
            np.testing.assert_allclose(uv,-concentric_uv(-coords,hole,1.2),atol=1e-6)

    def test_enlarged_offcenter_hole_render_preserves_actual_pupil(self):
        eye=displaced_eye([8,0]);LensTracker(static=True).update([eye],0)
        tex=material();tex.metadata={'calibration':{'hole':.32}}
        for scale in (1.,1.2,1.4):
            image=eye.crop.copy();compose_lens(image,eye,tex,.8,scale)
            np.testing.assert_array_equal(image[80,88],eye.crop[80,88])
            np.testing.assert_array_equal(image[eye.eyelid==0],eye.crop[eye.eyelid==0])
            self.assertTrue(np.any(image!=eye.crop))

if __name__=='__main__':unittest.main(verbosity=2)
