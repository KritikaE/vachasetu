import unittest
import numpy as np
import torch
import os
import shutil

from src.data_pipeline import (
    normalize_landmarks,
    generate_synthetic_lip_motion,
    LIP_LANDMARKS,
    NOSE_TIP_INDEX,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER
)
from src.model import LipCoordNet, ctc_decode, load_pretrained_model

class TestVachaSetuPipeline(unittest.TestCase):
    
    def setUp(self):
        class DummyLandmark:
            def __init__(self, x, y, z):
                self.x = x
                self.y = y
                self.z = z
        self.DummyLandmark = DummyLandmark

    def test_normalization_invariance(self):
        """
        Verify that coordinates normalization is invariant to translation, scale, and head tilt.
        """
        np.random.seed(42)
        base_face = np.random.normal(0, 10, size=(468, 3))
        
        base_face[NOSE_TIP_INDEX] = np.array([0.0, 0.0, 0.0])
        base_face[LEFT_EYE_OUTER] = np.array([-10.0, -5.0, 0.0])
        base_face[RIGHT_EYE_OUTER] = np.array([10.0, -5.0, 0.0])
        
        lms_base = [self.DummyLandmark(pt[0]/640.0, pt[1]/480.0, pt[2]/640.0) for pt in base_face]
        norm_base = normalize_landmarks(lms_base, width=640, height=480)
        
        trans_face = base_face + np.array([100.0, -50.0, 20.0])
        lms_trans = [self.DummyLandmark(pt[0]/640.0, pt[1]/480.0, pt[2]/640.0) for pt in trans_face]
        norm_trans = normalize_landmarks(lms_trans, width=640, height=480)
        
        angle = np.radians(10)
        c, s = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        rot_face = np.dot(base_face, rot_mat.T)
        lms_rot = [self.DummyLandmark(pt[0]/640.0, pt[1]/480.0, pt[2]/640.0) for pt in rot_face]
        norm_rot = normalize_landmarks(lms_rot, width=640, height=480)
        
        scaled_face = base_face * 2.5
        lms_scaled = [self.DummyLandmark(pt[0]/640.0, pt[1]/480.0, pt[2]/640.0) for pt in scaled_face]
        norm_scaled = normalize_landmarks(lms_scaled, width=640, height=480)
        
        self.assertEqual(norm_base.shape, (40, 3))
        np.testing.assert_allclose(norm_trans, norm_base, atol=1e-5)
        np.testing.assert_allclose(norm_rot, norm_base, atol=1e-5)
        np.testing.assert_allclose(norm_scaled, norm_base, atol=1e-5)

    def test_lipcoordnet_forward_pass(self):
        """
        Verify that LipCoordNet processes video tensors and coordinates successfully.
        """
        model = LipCoordNet(coord_input_dim=40, coord_hidden_dim=64)
        model.eval()
        
        # Mock inputs
        # Video: shape (B, C, T, H, W) -> (2, 3, 15, 64, 128)
        dummy_video = torch.randn(2, 3, 15, 64, 128)
        # Coordinates: shape (B, T, N, C) -> (2, 15, 20, 2)
        dummy_coords = torch.randn(2, 15, 20, 2)
        
        with torch.no_grad():
            outputs = model(dummy_video, dummy_coords)
            
        # Expected outputs shape: (B, T, 28) -> (2, 15, 28)
        self.assertEqual(outputs.shape, (2, 15, 28))

    def test_ctc_decoder(self):
        """
        Verify the CTC decoding mappings and output strings.
        """
        # Create a mock logit sequence: shape (T, 28) -> (5, 28)
        # Class index mappings: 0: blank, 1: space, 2: A, 3: B, 4: C
        logits = torch.zeros(7, 28)
        logits[0, 2] = 10.0   # A
        logits[1, 2] = 10.0   # A (duplicate, should collapse)
        logits[2, 0] = 10.0   # Blank
        logits[3, 3] = 10.0   # B
        logits[4, 1] = 10.0   # Space
        logits[5, 4] = 10.0   # C
        logits[6, 0] = 10.0   # Blank
        
        decoded_str, confidence = ctc_decode(logits)
        self.assertEqual(decoded_str, "AB C")
        self.assertGreater(confidence, 0.8)

if __name__ == "__main__":
    unittest.main()
