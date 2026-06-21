import os
import cv2
import numpy as np

def generate_simulated_video(output_path, text_label):
    """
    Generates a simulated lip-reading video by animating lips on top of assets/base_face.png.
    - sample1.mp4 simulates "Thank you" with slow open-close.
    - sample2.mp4 simulates "Help me" with faster wide/narrow lip motion.
    """
    print(f"Generating simulated video: {output_path} for '{text_label}'")
    
    # Path setup
    base_image_path = os.path.join("assets", "base_face.png")
    if not os.path.exists(base_image_path):
        # Fallback: create a purple face-like canvas
        print("Base face image not found. Creating a synthetic canvas.")
        base_img = np.zeros((480, 480, 3), dtype=np.uint8)
        # Fill with deep purple background
        base_img[:] = (20, 3, 11)  # BGR representation of #0B0314
        
        # Draw head outline
        cv2.ellipse(base_img, (240, 240), (160, 200), 0, 0, 360, (60, 10, 80), 2)
        # Draw eyes
        cv2.circle(base_img, (180, 180), 15, (157, 78, 221), 2)
        cv2.circle(base_img, (300, 180), 15, (157, 78, 221), 2)
        # Draw nose
        cv2.line(base_img, (240, 200), (240, 260), (212, 175, 55), 2)
        mouth_center = (240, 330)
        eye_dist = 120.0
    else:
        base_img = cv2.imread(base_image_path)
        height, width, _ = base_img.shape
        # Use simple heuristics to estimate mouth location if MediaPipe isn't running here
        # Normally mouth is around lower-middle center
        mouth_center = (int(width / 2), int(height * 0.72))
        eye_dist = width * 0.22

    height, width, _ = base_img.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 25.0, (width, height))
    
    # 50 frames at 25 fps = 2 seconds
    num_frames = 50
    
    for f in range(num_frames):
        frame_img = base_img.copy()
        
        # Progress from 0 to 1
        progress = f / float(num_frames)
        # Sine envelope for opening/closing motion
        envelope = np.sin(progress * np.pi)
        
        # Calculate dynamic lip parameters based on speech simulation
        if text_label == "Thank you":
            # Standard open-close
            open_factor = 1.0 + 1.5 * envelope
            stretch_factor = 1.0 + 0.2 * envelope
        else:  # "Help me"
            # Faster double peak or wider opening
            open_factor = 1.0 + 2.0 * np.sin(progress * 2 * np.pi)
            stretch_factor = 1.0 - 0.2 * envelope
            
        # Ensure open_factor doesn't go negative
        open_factor = max(0.1, open_factor)
        
        # Generate lip coordinates
        # Outer lips (20 points)
        outer_pts = []
        for i in range(20):
            theta = i * 2 * np.pi / 20
            # Width and height of lip ellipse
            rx = eye_dist * 0.4 * stretch_factor
            ry = eye_dist * 0.15 * open_factor
            
            # Simple distortion to make it look like lips
            if i > 5 and i < 15: # Lower lip
                ry *= 1.2
            
            x = int(mouth_center[0] + rx * np.cos(theta))
            y = int(mouth_center[1] + ry * np.sin(theta))
            outer_pts.append((x, y))
            
        # Inner lips (20 points)
        inner_pts = []
        for i in range(20):
            theta = i * 2 * np.pi / 20
            rx = eye_dist * 0.25 * stretch_factor
            ry = eye_dist * 0.08 * open_factor
            
            x = int(mouth_center[0] + rx * np.cos(theta))
            y = int(mouth_center[1] + ry * np.sin(theta))
            inner_pts.append((x, y))
            
        # Draw dark cover patch under lips to mask original mouth
        outer_pts_np = np.array(outer_pts, dtype=np.int32)
        cv2.fillPoly(frame_img, [outer_pts_np], (36, 20, 19))
        cv2.polylines(frame_img, [outer_pts_np], True, (157, 78, 221), 1)
        
        # Draw animated lips
        cv2.polylines(frame_img, [outer_pts_np], True, (255, 42, 116), 2)  # Neon Pink outer lip
        
        inner_pts_np = np.array(inner_pts, dtype=np.int32)
        cv2.polylines(frame_img, [inner_pts_np], True, (157, 78, 221), 1)  # Glowing Violet inner lip
        
        # Draw tracker dots on top
        for pt in outer_pts + inner_pts:
            cv2.circle(frame_img, pt, 2, (255, 255, 255), -1)
            
        out.write(frame_img)
        
    out.release()
    print(f"Finished generating {output_path}")

def main():
    print("=== Generating Preset Demo Videos Offline ===")
    output_dir = os.path.join("assets", "demo_videos")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate exactly two files
    generate_simulated_video(os.path.join(output_dir, "sample1.mp4"), "Thank you")
    generate_simulated_video(os.path.join(output_dir, "sample2.mp4"), "Help me")
    
    print("Demo videos acquisition task complete.")

if __name__ == "__main__":
    main()
