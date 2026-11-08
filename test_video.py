"""Offline input ordering, media-time tracking, playback and orientation regressions."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from video_app import VideoReader,VideoSession,VideoControls
from lens_import import LensImportControl
from eye_pipeline import ROOT


class FakePipeline:
    def __init__(self):self.timestamps=[];self.resets=0
    def reset_tracking(self):self.resets+=1
    def process(self,frame,timestamp=None):
        self.timestamps.append(timestamp)
        return [],{'mesh_ms':0.,'iris_ms':0.,'total_ms':0.}


class VideoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.path=Path(cls.temp.name)/'离线测试.avi'
        writer=cv2.VideoWriter(str(cls.path),cv2.VideoWriter_fourcc(*'MJPG'),10,(64,96))
        if not writer.isOpened():raise RuntimeError('MJPG test fixture encoder unavailable')
        for i in range(12):writer.write(np.full((96,64,3),i*18,np.uint8))
        writer.release()
    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_decode_all_frames_in_order_and_keep_portrait(self):
        reader=VideoReader(self.path)
        try:
            self.assertEqual((reader.width,reader.height),(64,96))
            frames=[reader.first]
            while True:
                item=reader.read()
                if item is None:break
                frames.append(item)
            self.assertEqual([f[2] for f in frames],list(range(12)))
            np.testing.assert_allclose([f[1] for f in frames],np.arange(12)/10,atol=.001)
            self.assertTrue(all(frames[i+1][0].mean()>frames[i][0].mean() for i in range(11)))
        finally:reader.close()

    def test_pause_end_replay_and_media_time(self):
        pipeline=FakePipeline();session=VideoSession(pipeline)
        try:
            session.load(self.path);session.paused=True
            self.assertFalse(session.advance());self.assertEqual(session.index,0)
            self.assertEqual(len(pipeline.timestamps),1)
            session.paused=False
            while session.advance():pass
            self.assertTrue(session.ended);self.assertTrue(session.paused)
            self.assertEqual(session.index,11)
            session.restart()
            self.assertEqual(session.index,0);self.assertFalse(session.ended);self.assertFalse(session.paused)
            self.assertEqual(pipeline.timestamps[-1],0.)
            self.assertEqual(pipeline.resets,2)
        finally:session.close()

    def test_seek_and_failed_replace_preserve_current_source(self):
        pipeline=FakePipeline();session=VideoSession(pipeline)
        try:
            session.load(self.path);session.paused=True;session.seek(7)
            self.assertEqual(session.index,7);self.assertTrue(session.paused)
            self.assertAlmostEqual(pipeline.timestamps[-1],.7)
            existing=session.reader
            with self.assertRaises(FileNotFoundError):session.load(Path(self.temp.name)/'missing.mp4')
            self.assertIs(session.reader,existing);self.assertEqual(session.index,7)
            session.paused=False;self.assertTrue(session.advance());self.assertEqual(session.index,8)
        finally:session.close()

    def test_import_button_cancel_and_dispatch(self):
        control=LensImportControl(state_file=Path(self.temp.name)/'selection.json')
        control.decorate(np.zeros((752,1920,3),np.uint8))
        called=[];control.video_callback=called.append
        x0,y0,x1,y1=control.rects['video']
        control.on_mouse(cv2.EVENT_LBUTTONUP,(x0+x1)//2,(y0+y1)//2,0,None)
        with patch('video_app.choose_video',return_value=self.path):self.assertTrue(control.handle_pending())
        self.assertEqual(called,[self.path])
        with patch('video_app.choose_video',return_value=None):self.assertFalse(control.handle_pending(ord('v')))
        self.assertEqual(called,[self.path])

    def test_controls_seek_and_image_bounds(self):
        session=VideoSession(FakePipeline());session.load(self.path)
        try:
            controls=VideoControls()
            output=controls.decorate(np.zeros((1080,1920,3),np.uint8),session,True,30)
            self.assertEqual(output.shape,(1080,1920,3))
            x0,y0,x1,y1=controls.rects['seek']
            self.assertTrue(controls.on_mouse(cv2.EVENT_LBUTTONUP,(x0+x1)//2,(y0+y1)//2,0,None))
            self.assertEqual(controls.pending,('seek',.5))
        finally:session.close()

    def test_supplied_rotated_hevc_video(self):
        path=ROOT/'微信视频2026-09-21_015131_959.mp4'
        if not path.exists():self.skipTest('Optional user video not present')
        reader=VideoReader(path)
        try:
            self.assertEqual((reader.width,reader.height),(1080,1920))
            self.assertEqual(reader.first[0].shape[:2],(1920,1080))
            self.assertEqual(reader.frame_count,343)
        finally:reader.close()

if __name__=='__main__':unittest.main(verbosity=2)
