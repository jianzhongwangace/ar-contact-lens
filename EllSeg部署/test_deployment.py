"""Verify inference parity, transforms, constant inputs and result serialization."""
import ast
import json
import unittest
import cv2
import numpy as np
import torch
from ellseg_core import ROOT,WEIGHTS,EllSegPredictor,preprocess,transform_ellipse,soft_center
from vendor import ellseg_model

class GeometryTests(unittest.TestCase):
    def test_constant_image_normalization_has_no_nan(self):
        x,matrix=preprocess(np.full((160,256,3),80,np.uint8))
        self.assertEqual(x.shape,(1,1,240,320))
        self.assertTrue(np.isfinite(x).all())

    def test_preprocess_affine_roundtrip_for_portrait_and_landscape(self):
        for shape in [(100,300),(400,200),(240,320)]:
            _,matrix=preprocess(np.zeros((*shape,3),np.uint8))
            point=np.array([[[shape[1]/2,shape[0]/2]]],np.float32)
            actual=cv2.transform(cv2.transform(point,matrix),cv2.invertAffineTransform(matrix))
            np.testing.assert_allclose(point,actual,atol=1e-4)

    def test_affine_ellipse_preserves_points_under_nonsquare_scaling(self):
        params=np.array([.1,-.2,.5,.3,.6])
        affine=np.array([[160,0,160],[0,120,120]],float)
        transformed=transform_ellipse(params,affine)
        theta=np.linspace(0,2*np.pi,100)
        rotation=np.array([[np.cos(params[4]),-np.sin(params[4])],[np.sin(params[4]),np.cos(params[4])]])
        points=np.stack([params[2]*np.cos(theta),params[3]*np.sin(theta)],axis=1)@rotation.T+params[:2]
        points=points@affine[:,:2].T+affine[:,2]
        angle=transformed[4];rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        unit=(points-transformed[:2])@rot/transformed[2:4]
        np.testing.assert_allclose(np.sum(unit**2,axis=1),1,atol=1e-10)

    def test_soft_center_matches_upstream_torch_temperature_four(self):
        rng=np.random.default_rng(4);a=rng.normal(size=(24,32)).astype(np.float32)
        tensor=torch.from_numpy(a);weights=torch.softmax(tensor.reshape(-1)*4,dim=0)
        yy,xx=torch.meshgrid(torch.linspace(-1,1,24),torch.linspace(-1,1,32),indexing='ij')
        expected=np.array([(weights*xx.reshape(-1)).sum().item(),(weights*yy.reshape(-1)).sum().item()])
        np.testing.assert_allclose(soft_center(a),expected,atol=1e-6)

class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cpu=EllSegPredictor('cpu')

    def test_strict_weights_and_valid_shapes(self):
        logits,reg=self.cpu.run(np.zeros((1,1,240,320),np.float32))
        self.assertEqual(logits.shape,(1,3,240,320));self.assertEqual(reg.shape,(1,10))
        self.assertTrue(np.isfinite(logits).all());self.assertTrue(np.isfinite(reg).all())

    def test_adapted_blocks_match_original_network_operations(self):
        source=(ROOT/'upstream_reference/RITnet_v3.py').read_text(encoding='utf-8').replace('np.int(', 'int(')
        tree=ast.parse(source)
        # Evaluate only definitions; training-only imports are unnecessary for
        # the official evaluate_ellseg enc/dec/elReg forward path.
        tree.body=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))]
        namespace={'torch':torch,'np':np,'nn':torch.nn,'F':torch.nn.functional,
                   'convBlock':ellseg_model.convBlock,'regressionModule':ellseg_model.regressionModule}
        exec(compile(tree,'upstream-reference','exec'),namespace)
        original=namespace['DenseNet2D']()
        original.load_state_dict(torch.load(WEIGHTS,map_location='cpu',weights_only=True)['state_dict'],strict=True)
        original.eval()
        torch.manual_seed(53);x=torch.randn(1,1,240,320)
        with torch.inference_mode():
            a,b,c,d,z=original.enc(x)
            expected_seg=original.dec(a,b,c,d,z)
            expected_reg=original.elReg(z,0)
            actual_seg,actual_reg=self.cpu.model(x)
        torch.testing.assert_close(actual_seg,expected_seg,atol=0,rtol=0)
        torch.testing.assert_close(actual_reg,expected_reg,atol=0,rtol=0)

    def test_gpu_matches_cpu_on_eye_and_constant_images(self):
        import onnxruntime as ort
        if 'DmlExecutionProvider' not in ort.get_available_providers():self.skipTest('DirectML unavailable')
        gpu=EllSegPredictor('dml')
        fixture=cv2.imdecode(np.fromfile(str(ROOT/'test_assets/eye.png'),np.uint8),1)
        self.assertIsNotNone(fixture)
        results=[]
        for name,image in [('eye',fixture),('dark',(fixture.astype(float)*.6).astype(np.uint8)),('blank',np.zeros_like(fixture))]:
            x,_=preprocess(image);cpu_seg,cpu_reg=self.cpu.run(x);gpu_seg,gpu_reg=gpu.run(x)
            err=float(np.max(np.abs(cpu_seg-gpu_seg)));reg_err=float(np.max(np.abs(cpu_reg-gpu_reg)))
            disagreement=float(np.mean(cpu_seg.argmax(1)!=gpu_seg.argmax(1)))
            self.assertTrue(np.isfinite(gpu_seg).all());self.assertTrue(np.isfinite(gpu_reg).all())
            if name=='blank':
                # Raw constant-input logits are backend-sensitive because of
                # InstanceNorm. The product must suppress such invalid input.
                for predictor in (self.cpu,gpu):
                    result=predictor.predict(image)
                    self.assertFalse(result['input_valid']);self.assertFalse(result['mask'].any())
                    self.assertFalse(any(r['plausible'] for r in result['ellipses'].values()))
            else:
                self.assertLess(err,.005);self.assertLess(reg_err,.001);self.assertLess(disagreement,.001)
            results.append({'case':name,'logit_max_error':err,'regression_max_error':reg_err,
                            'label_disagreement':disagreement,'invalid_input_suppressed':name=='blank'})
        (ROOT/'outputs/gpu-parity.json').write_text(json.dumps({'adapter':gpu.adapter,'comparisons':results},indent=2))

    def test_original_image_size_and_json_output(self):
        image=cv2.imdecode(np.fromfile(str(ROOT/'test_assets/eye.png'),np.uint8),1)
        result=self.cpu.predict(image)
        self.assertEqual(result['mask'].shape,image.shape[:2])
        self.assertTrue(set(np.unique(result['mask'])).issubset({0,1,2}))
        json.dumps(result['ellipses'],allow_nan=False)
        for record in result['ellipses'].values():self.assertTrue(np.isfinite(record['ellipse']).all())

if __name__=='__main__':unittest.main(verbosity=2)
