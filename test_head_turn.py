"""Verified motion must follow current pixels and must not bridge blinks forever."""
import unittest
import cv2
import numpy as np
from lens_renderer import LensTracker
from test_lens_rendering import make_eye
from turn_tracking import verify_eye_motion


def textured_eye():
    eye=make_eye()
    rng=np.random.default_rng(43)
    noise=rng.integers(25,180,eye.crop.shape[:2],dtype=np.uint8)
    gray=cv2.GaussianBlur(noise,(3,3),.5)
    eye.crop=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    return eye


class MotionTests(unittest.TestCase):
    def test_current_translation_verified(self):
        e=textured_eye();tracker=LensTracker();tracker.update([e],1.)
        prior=tracker.history[e.side]
        transform=np.array([[1.,0,3],[0,1,1]])
        e.crop=cv2.warpAffine(e.crop,transform,(160,160))
        e.center+=[3,1];e.iris_points+=[3,1]
        self.assertTrue(verify_eye_motion(prior,e)[0])
        e.center+=[30,0]
        self.assertFalse(verify_eye_motion(prior,e)[0])

    def test_flat_or_changed_occluder_rejected(self):
        e=textured_eye();tracker=LensTracker();tracker.update([e],1.)
        prior=tracker.history[e.side]
        e.crop[:]=80
        self.assertFalse(verify_eye_motion(prior,e)[0])
        e.crop=np.random.default_rng(77).integers(20,190,e.crop.shape,dtype=np.uint8)
        self.assertFalse(verify_eye_motion(prior,e)[0])

    def test_verified_gap_expires_without_refreshing_evidence(self):
        e=textured_eye();tracker=LensTracker();tracker.update([e],1.)
        e.iris[:]=0;e.iris_prob[:]=0
        for now in [1.05,1.15,1.25,1.35]:
            tracker.update([e],now)
            self.assertEqual(e.tracking_state,'verified motion')
            self.assertEqual(tracker.history[e.side]['reliable_time'],1.)
        tracker.update([e],1.46)
        self.assertIsNone(e.lens_pose)

    def test_closed_eye_and_missing_face_reset_verified_gap(self):
        e=textured_eye();tracker=LensTracker();tracker.update([e],1.)
        e.iris[:]=0;tracker.update([e],1.05)
        e.is_open=False;tracker.update([e],1.10)
        self.assertIsNone(e.lens_pose);self.assertFalse(tracker.history)
        e=textured_eye();tracker.update([e],2.)
        tracker.update([],2.05);self.assertFalse(tracker.history)


if __name__=='__main__':unittest.main()
