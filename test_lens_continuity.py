"""Regression scenes for blink/dropout handling and broken segmentation masks."""
import unittest
import cv2
import numpy as np
from lens_renderer import LensTracker,compose_lens
from test_lens_rendering import make_eye,material


class ContinuityTests(unittest.TestCase):
    def test_short_segmentation_dropout_uses_current_geometry_then_expires(self):
        e=make_eye();tracker=LensTracker();tracker.update([e],1.)
        old=e.lens_pose.copy()
        e.iris[:]=0;e.center[0]+=3;e.iris_points[:,0]+=3
        tracker.update([e],1.033)
        self.assertEqual(e.tracking_state,'brief fallback')
        self.assertIsNotNone(e.lens_pose)
        # Unified-center mode may hold the last reliable pupil center while the
        # iris-only geometry is temporarily missing; it must remain present and
        # bounded rather than jumping to the incomplete mask.
        self.assertLess(abs(e.lens_pose[0]-old[0]),3.0)
        self.assertEqual(e.lens_opacity,1.)
        tracker.update([e],1.16)
        self.assertGreater(e.lens_opacity,0.)
        self.assertLess(e.lens_opacity,1.)
        tracker.update([e],1.24)
        self.assertIsNone(e.lens_pose)
        self.assertFalse(tracker.history)

    def test_dropouts_never_initialize_unverified_eye(self):
        e=make_eye();e.iris[:]=0
        tracker=LensTracker();tracker.update([e],1.)
        self.assertIsNone(e.lens_pose)

    def test_blink_and_missing_face_hide_immediately_even_during_fallback(self):
        e=make_eye();tracker=LensTracker();tracker.update([e],1.)
        e.iris[:]=0;tracker.update([e],1.03)
        self.assertIsNotNone(e.lens_pose)
        e.is_open=False;tracker.update([e],1.06)
        self.assertIsNone(e.lens_pose);self.assertFalse(tracker.history)
        e=make_eye();tracker.update([e],2.);tracker.update([],2.03)
        self.assertFalse(tracker.history)

    def test_large_position_jump_cannot_reuse_old_pose(self):
        e=make_eye();tracker=LensTracker();tracker.update([e],1.)
        e.iris[:]=0;e.affine[:,2]-=[100,0]
        tracker.update([e],1.03)
        self.assertIsNone(e.lens_pose)

    def test_broken_iris_mask_does_not_cut_texture_holes(self):
        e=make_eye();full=e.crop.copy()
        compose_lens(full,e,material(),.8)
        # Remove evidence in BOTH the diagnostic mask and the network map.
        # Internal cracks must not punch a matching hole in the texture.
        for source in (e.iris,e.iris_prob):
            source[70:88,99:108]=0
            source[68:72,43:49]=0
        LensTracker(static=True).update([e])
        self.assertIsNotNone(e.lens_pose)
        broken=e.crop.copy();compose_lens(broken,e,material(),.8)
        np.testing.assert_array_equal(full,broken)
        self.assertTrue(np.any(broken[70:92,101:113]!=e.crop[70:92,101:113]))

    def test_missing_outer_half_does_not_paint_full_ellipse_on_skin(self):
        e=make_eye()
        e.iris[:,83:]=0;e.iris_prob[:,83:]=0
        LensTracker(static=True).update([e])
        self.assertIsNotNone(e.lens_pose)
        rendered=e.crop.copy();compose_lens(rendered,e,material(),.8)
        np.testing.assert_array_equal(rendered[:,88:],e.crop[:,88:])
        self.assertTrue(np.any(rendered[:,:80]!=e.crop[:,:80]))

    def test_huge_false_pupil_does_not_erase_the_lens(self):
        e=make_eye()
        e.pupil_prob=(e.iris>0).astype(np.float32)
        e.pupil=e.iris.copy()
        LensTracker(static=True).update([e])
        self.assertIsNotNone(e.lens_pose)
        self.assertFalse(e.pupil_protection.any())
        rendered=e.crop.copy();compose_lens(rendered,e,material())
        self.assertTrue(np.any(rendered!=e.crop))

    def test_static_images_do_not_borrow_history(self):
        e=make_eye();tracker=LensTracker(static=True);tracker.update([e],1.)
        e.iris[:]=0;tracker.update([e],1.03)
        self.assertIsNone(e.lens_pose)

    def test_pupil_fallback_is_time_bounded(self):
        e=make_eye();tracker=LensTracker();tracker.update([e],1.)
        self.assertTrue(e.pupil_protection.any())
        e.pupil_prob[:]=0
        tracker.update([e],1.03);self.assertTrue(e.pupil_protection.any())
        for t in [1.10,1.18,1.26]:tracker.update([e],t)
        self.assertFalse(e.pupil_protection.any())

if __name__=='__main__':unittest.main(verbosity=2)
