import numpy as np
import torch

# Indices for outer and inner lip boundaries in MediaPipe Face Mesh
# Outer Lip: 20 landmarks
OUTER_LIP_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
# Inner Lip: 20 landmarks
INNER_LIP_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

LIP_LANDMARKS = OUTER_LIP_INDICES + INNER_LIP_INDICES  # 40 landmarks in total

# Anchor landmarks for normalization
NOSE_TIP_INDEX = 4
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

VOCABULARY = ["Help", "Yes", "No", "Thank You", "Water"]

def normalize_landmarks(raw_coords, width, height):
    """
    Normalizes a set of face landmarks to be invariant to translation, scale, and minor head tilt.
    
    Parameters:
    - raw_coords: dict or list of raw landmark objects/tuples containing .x, .y, .z
    - width: width of the frame
    - height: height of the frame
    
    Returns:
    - Normalized numpy array of shape (40, 3) representing the lip landmarks.
    """
    # Convert all MediaPipe landmarks to pixel space (isotropic scaling)
    # We use width for z scaling as standard behavior to match x scale.
    coords_px = np.array([[l.x * width, l.y * height, l.z * width] for l in raw_coords])

    # Extract anchor coordinates
    p_nose = coords_px[NOSE_TIP_INDEX]
    p_eye_l = coords_px[LEFT_EYE_OUTER]
    p_eye_r = coords_px[RIGHT_EYE_OUTER]

    # Translate all coordinates relative to the nose tip (nose tip becomes 0,0,0)
    translated = coords_px - p_nose

    # Calculate eye vector in 2D (x, y) to find roll (head tilt)
    # We want to align this vector horizontally
    eye_vec = p_eye_r[:2] - p_eye_l[:2]
    theta = np.arctan2(eye_vec[1], eye_vec[0])  # Angle in radians

    # 2D Rotation Matrix to rotate by -theta
    c, s = np.cos(-theta), np.sin(-theta)
    rot_matrix = np.array([
        [c, -s],
        [s, c]
    ])

    # Rotate x, y coordinates
    rotated_xy = np.dot(translated[:, :2], rot_matrix.T)
    
    # Reassemble 3D coordinates (z remains unchanged by roll)
    rotated = np.zeros_like(translated)
    rotated[:, :2] = rotated_xy
    rotated[:, 2] = translated[:, 2]

    # Calculate scale factor: distance between the eye corners
    # (Distance is invariant under translation and rotation)
    scale_dist = np.linalg.norm(p_eye_r[:2] - p_eye_l[:2])
    if scale_dist < 1e-5:
        scale_dist = 1.0  # Avoid division by zero

    # Scale the rotated coordinates
    normalized = rotated / scale_dist

    # Extract only the lip landmarks (shape: (40, 3))
    lip_normalized = normalized[LIP_LANDMARKS]

    return lip_normalized

class LandmarkBuffer:
    """
    Maintains a sliding window buffer of lip landmarks for real-time inference.
    """
    def __init__(self, window_size=30, num_features=120):
        self.window_size = window_size
        self.num_features = num_features
        self.buffer = []

    def append(self, landmarks):
        """
        Append a single frame of landmarks.
        landmarks should be a flattened numpy array of shape (120,)
        """
        self.buffer.append(landmarks)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def clear(self):
        self.buffer.clear()

    def is_ready(self):
        return len(self.buffer) > 0

    def get_tensor(self):
        """
        Returns a PyTorch tensor of shape (1, window_size, num_features)
        with replication padding if the buffer is not yet full.
        """
        if not self.buffer:
            return torch.zeros((1, self.window_size, self.num_features), dtype=torch.float32)

        current_len = len(self.buffer)
        if current_len >= self.window_size:
            data = np.array(self.buffer[-self.window_size:])
        else:
            # Replicate the oldest frame to pad
            pad_len = self.window_size - current_len
            pad = [self.buffer[0]] * pad_len
            data = np.array(pad + self.buffer)

        # Reshape to (1, window_size, num_features)
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0)

def generate_synthetic_lip_motion(label_idx, num_frames=30, add_noise=True):
    """
    Generates a synthetic sequence of 40 normalized lip landmarks (x, y, z) 
    representing one of the vocabulary classes.
    """
    # 1. Define base shape of outer and inner lips in normalized space
    # Outer lip is an ellipse, inner lip is a smaller ellipse
    t_outer = np.linspace(0, 2*np.pi, 20, endpoint=False)
    t_inner = np.linspace(0, 2*np.pi, 20, endpoint=False)

    base_outer_x = 0.4 * np.cos(t_outer)
    base_outer_y = 0.2 * np.sin(t_outer)
    base_inner_x = 0.25 * np.cos(t_inner)
    base_inner_y = 0.1 * np.sin(t_inner)

    base_x = np.concatenate([base_outer_x, base_inner_x])
    base_y = np.concatenate([base_outer_y, base_inner_y])
    base_z = np.zeros(40)  # Flat lips on Z-axis initially

    sequence = []
    
    # We will simulate mouth movements over time using a sine wave envelope
    for f in range(num_frames):
        # Progress from 0 to 1
        progress = f / float(num_frames)
        # Sine envelope for opening/closing motion
        envelope = np.sin(progress * np.pi)  # 0 at start/end, peak at middle

        # Copy base coordinates
        x = base_x.copy()
        y = base_y.copy()
        z = base_z.copy()

        # Class-specific deformations
        if label_idx == 0:  # "Help" - standard open/close twice or deep open
            # Modulate vertical opening (y-coords of upper/lower parts)
            # Upper lip indices have negative y, lower have positive y (or vice-versa depending on orientation)
            # Let's say we stretch Y coordinates
            y *= (1.0 + 1.2 * envelope)
        elif label_idx == 1:  # "Yes" - wide horizontal smile, slight open
            x *= (1.0 + 0.3 * envelope)
            y *= (1.0 + 0.4 * envelope)
        elif label_idx == 2:  # "No" - tight rounded mouth, narrow width
            x *= (1.0 - 0.3 * envelope)
            y *= (1.0 + 0.6 * envelope)
        elif label_idx == 3:  # "Thank You" - closed, then wide open, then pursed
            if progress < 0.3:
                y *= 0.2  # Closed/pursed
            elif progress < 0.7:
                y *= 1.5  # Open
            else:
                x *= 0.6  # Pursed
                y *= 0.8
        elif label_idx == 4:  # "Water" - pursed circle, then opening wide, then closing
            if progress < 0.4:
                x *= 0.5
                y *= 1.2
            else:
                x *= 1.2
                y *= 0.7

        # Reassemble to shape (40, 3)
        frame_coords = np.stack([x, y, z], axis=1)

        # If add_noise is true, we simulate translation, rotation, scaling to mimic raw inputs
        # that our normalization pipeline must correct.
        if add_noise:
            # Add translation
            trans = np.random.uniform(-50, 50, size=3)
            # Add rotation (roll angle between -15 and 15 degrees)
            angle = np.random.uniform(-np.radians(15), np.radians(15))
            c, s = np.cos(angle), np.sin(angle)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            # Add scale factor (simulate camera distance)
            scale = np.random.uniform(100.0, 300.0)

            # Apply scale, rotation, translation
            frame_coords = np.dot(frame_coords * scale, rot.T) + trans

            # Generate mock "nose tip" (4) and "eye corners" (33, 263) so normalize_landmarks can run on it.
            # We create a dummy full face landmark list of 468 points.
            full_face = np.random.normal(0, 10, size=(468, 3))
            
            # Place nose tip at trans (since we translated by trans)
            full_face[NOSE_TIP_INDEX] = trans
            
            # Place eyes around the nose
            eye_l_raw = np.array([-0.5, -0.4, 0.0]) * scale
            eye_r_raw = np.array([0.5, -0.4, 0.0]) * scale
            full_face[LEFT_EYE_OUTER] = np.dot(eye_l_raw, rot.T) + trans
            full_face[RIGHT_EYE_OUTER] = np.dot(eye_r_raw, rot.T) + trans

            # Inject the transformed lip landmarks into full_face
            full_face[LIP_LANDMARKS] = frame_coords

            # Run normalization to see if it cleans it up!
            # We pass width=640, height=480, and raw MediaPipe-like objects.
            # To simulate landmarks, we wrap them in a simple namespace.
            class DummyLandmark:
                def __init__(self, pt):
                    # divide by width/height to simulate raw normalized landmarks
                    self.x = pt[0] / 640.0
                    self.y = pt[1] / 480.0
                    self.z = pt[2] / 640.0

            dummy_lms = [DummyLandmark(pt) for pt in full_face]
            frame_coords = normalize_landmarks(dummy_lms, width=640, height=480)

        # Flatten frame to (120,)
        sequence.append(frame_coords.flatten())

    return np.array(sequence)

def generate_synthetic_dataset(num_samples_per_class=100, num_frames=30):
    """
    Generates a full synthetic dataset for training and verification.
    
    Returns:
    - X: numpy array of shape (num_classes * num_samples, num_frames, 120)
    - y: numpy array of shape (num_classes * num_samples,)
    """
    X = []
    y = []
    
    for idx in range(len(VOCABULARY)):
        for _ in range(num_samples_per_class):
            seq = generate_synthetic_lip_motion(idx, num_frames=num_frames, add_noise=True)
            X.append(seq)
            y.append(idx)
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)
