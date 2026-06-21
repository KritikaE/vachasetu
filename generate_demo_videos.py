import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.data_pipeline import (
    generate_synthetic_lip_motion,
    VOCABULARY,
    LIP_LANDMARKS,
    NOSE_TIP_INDEX,
    LEFT_EYE_OUTER,
    RIGHT_EYE_OUTER,
    OUTER_LIP_INDICES,
    INNER_LIP_INDICES
)

def main():
    print("=== Generating Preset Demo Videos ===")
    
    # Paths
    base_image_path = "assets/base_face.png"
    output_dir = "assets/videos"
    model_path = "face_landmarker.task"
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(base_image_path):
        print(f"Error: Base face portrait not found at {base_image_path}")
        return
        
    if not os.path.exists(model_path):
        print("Downloading Face Landmarker model...")
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete.")
        
    # Load base image
    base_img = cv2.imread(base_image_path)
    height, width, _ = base_img.shape
    print(f"Loaded base portrait: {width}x{height}")
    
    # Initialize MediaPipe Face Landmarker
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1
    )
    
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        # Process image to find anchors
        rgb_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
        results = landmarker.detect(mp_img)
        
        if not results.face_landmarks:
            print("Error: MediaPipe could not detect a face in the base portrait image.")
            print("Please check that the image is a clear, front-facing face.")
            print("Using fallback coordinate anchors...")
            p_nose = np.array([width / 2.0, height / 2.0 + 30.0, 0.0])
            p_eye_l = np.array([width / 2.0 - 60.0, height / 2.0 - 50.0, 0.0])
            p_eye_r = np.array([width / 2.0 + 60.0, height / 2.0 - 50.0, 0.0])
            lip_lms_px = np.array([[width/2.0 + 40.0 * np.cos(t), height/2.0 + 40.0 + 20.0 * np.sin(t), 0.0] for t in np.linspace(0, 2*np.pi, 40)])
        else:
            print("Face successfully detected by MediaPipe Face Landmarker.")
            landmarks = results.face_landmarks[0]
            
            # Convert landmarks to pixel space
            coords_px = np.array([[l.x * width, l.y * height, l.z * width] for l in landmarks])
            p_nose = coords_px[NOSE_TIP_INDEX]
            p_eye_l = coords_px[LEFT_EYE_OUTER]
            p_eye_r = coords_px[RIGHT_EYE_OUTER]
            lip_lms_px = coords_px[LIP_LANDMARKS]
            
        # Calculate scale and translation anchors
        eye_dist = np.linalg.norm(p_eye_r[:2] - p_eye_l[:2])
        mouth_center = np.mean(lip_lms_px[:, :2], axis=0)
        
        # Find bounding polygon for the mouth to draw the dark cover patch
        outer_lip_pts = lip_lms_px[:20, :2]
        mouth_hull = cv2.convexHull(outer_lip_pts.astype(np.int32))
        
        # 4-character code for MP4 encoding
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Generate a video for each word in the vocabulary
        for idx, word in enumerate(VOCABULARY):
            print(f"Generating video for word: '{word}'...")
            
            video_filename = f"demo_{word.lower().replace(' ', '_')}.mp4"
            video_path = os.path.join(output_dir, video_filename)
            
            out = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
            synth_seq = generate_synthetic_lip_motion(idx, num_frames=30, add_noise=False)
            
            for frame_idx in range(30):
                frame_img = base_img.copy()
                
                # 1. Mask out the original mouth with a dark purple digital patch
                cv2.fillPoly(frame_img, [mouth_hull], (36, 20, 19))
                cv2.polylines(frame_img, [mouth_hull], True, (157, 78, 221), 1)
                
                # 2. Extract normalized coordinates for this frame (shape: (40, 3))
                coords = synth_seq[frame_idx].reshape(40, 3)
                
                # Map coordinates back to mouth region
                mapped_pts = []
                for pt in coords:
                    x_px = int(mouth_center[0] + pt[0] * eye_dist * 1.8)
                    y_px = int(mouth_center[1] + pt[1] * eye_dist * 1.8)
                    mapped_pts.append((x_px, y_px))
                    
                # 3. Draw moving lip loops
                outer_pts = np.array(mapped_pts[:20], dtype=np.int32)
                cv2.polylines(frame_img, [outer_pts], True, (255, 42, 116), 2)  # Neon Pink / Magenta
                
                inner_pts = np.array(mapped_pts[20:], dtype=np.int32)
                cv2.polylines(frame_img, [inner_pts], True, (157, 78, 221), 1)  # Glowing Violet
                
                # Draw little neon landmark tracker circles
                for pt in mapped_pts:
                    cv2.circle(frame_img, pt, 2, (255, 255, 255), -1)  # White tracker dots
                    
                out.write(frame_img)
                
            out.release()
            print(f"Saved: {video_path}")
            
    print("All preset demo videos generated successfully!")

if __name__ == "__main__":
    main()
