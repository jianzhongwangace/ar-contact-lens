"""Static-noise suppression must not freeze real motion or delay occlusion."""
import unittest
import cv2
import numpy as np
from lens_renderer import LensTracker,EyeViewTracker,stabilize_pose,stabilize_support,compose_lens
from test_lens_rendering import make_eye,material


class StabilityTests(unittest.TestCase):
    def test_stationary_position_axes_and_roll_reject_measurement_noise(self):
        truth=np.array([200.,150.,20.,18.,.1])
        state={'pose':truth.copy(),'observations':[truth.copy()]}
        rng=np.random.default_rng(20260921)
        raw_samples=[];stable_samples=[]
        for i in range(180):
            raw=truth+rng.normal(0,[.65,.65,.55,.5,.025])
            stable,observations=stabilize_pose(raw,state,1/30)
            state={'pose':stable,'observations':observations}
            if i>20:raw_samples.append(raw);stable_samples.append(stable)
        before=np.std(raw_samples,axis=0);after=np.std(stable_samples,axis=0)
        self.assertTrue(np.all(after<before*.45),(before,after))

    def test_single_frame_outlier_does_not_move_lens(self):
        truth=np.array([100.,100.,20.,18.,0.])
        prior={'pose':truth.copy(),'observations':[truth.copy(),truth.copy()]}
        measured=truth+np.array([5.,-4.,2.,-1.,.1])
        stable,_=stabilize_pose(measured,prior,1/30)
        np.testing.assert_array_equal(stable,truth)

    def test_confirmed_movement_follows_within_two_frames(self):
        start=np.array([100.,100.,20.,18.,0.]);target=start.copy();target[0]+=8
        state={'pose':start.copy(),'observations':[start.copy(),start.copy()]}
        for _ in range(2):
            stable,observations=stabilize_pose(target,state,1/30)
            state={'pose':stable,'observations':observations}
        self.assertLess(np.linalg.norm(stable[:2]-target[:2]),.8)

    def test_slow_motion_does_not_stick_permanently(self):
        truth=np.array([100.,100.,20.,18.,0.])
        state={'pose':truth.copy(),'observations':[truth.copy()]}
        for i in range(1,61):
            moving=truth.copy();moving[0]+=.35*i
            stable,observations=stabilize_pose(moving,state,1/30)
            state={'pose':stable,'observations':observations}
        self.assertLess(abs(stable[0]-moving[0]),2.5)

    def test_small_mask_boundary_noise_is_smoothed(self):
        first=np.zeros((128,128),np.float32);cv2.circle(first,(64,64),50,1,-1)
        noise=first.copy();noise[:,112:]=0
        stable=stabilize_support(noise,first,1/30)
        self.assertLess(np.abs(stable-first).sum(),np.abs(noise-first).sum())

    def test_large_new_occlusion_is_not_averaged_with_previous_mask(self):
        first=np.zeros((128,128),np.float32);cv2.circle(first,(64,64),50,1,-1)
        current=first.copy();current[64:]=0
        stable=stabilize_support(current,first,1/30)
        self.assertFalse(stable[64:].any())

    def test_real_eyelid_always_clips_stabilized_masks(self):
        eye=make_eye();tracker=LensTracker();tracker.update([eye],1.)
        eye.eyelid[80:]=0
        tracker.update([eye],1.033)
        frame=eye.crop.copy();expected=frame.copy()
        compose_lens(frame,eye,material())
        np.testing.assert_array_equal(frame[eye.eyelid==0],expected[eye.eyelid==0])
        eye.is_open=False;tracker.update([eye],1.066)
        self.assertIsNone(eye.lens_pose)
        self.assertFalse(tracker.history)

    def test_magnified_view_stable_without_changing_inference_crop(self):
        tracker=EyeViewTracker();rng=np.random.default_rng(61)
        measured=[];displayed=[]
        for i in range(100):
            eye=make_eye()
            eye.affine[:,2]=rng.normal(0,.65,2)
            original_affine=eye.affine.copy();original_crop=eye.crop.copy()
            tracker.update([eye],i/30)
            np.testing.assert_array_equal(eye.affine,original_affine)
            np.testing.assert_array_equal(eye.crop,original_crop)
            if i>20:
                measured.append(eye.affine[:,2]);displayed.append(eye.display_affine[:,2])
        self.assertLess(float(np.std(displayed,axis=0).mean()),float(np.std(measured,axis=0).mean())*.45)
        tracker.update([],4.);self.assertFalse(tracker.history)

    def test_static_display_crop_is_unchanged(self):
        eye=make_eye();eye.affine=cv2.getRotationMatrix2D((80,80),12,.9)
        EyeViewTracker(static=True).update([eye],0)
        np.testing.assert_allclose(eye.display_affine,eye.affine,atol=1e-9)

if __name__=='__main__':unittest.main(verbosity=2)
