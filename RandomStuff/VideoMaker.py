import cv2
import os
import natsort

# -----------------------------
# User settings
# -----------------------------
image_folder = r"C:\Users\emcof\JasonTest_files\dp0\FFF-1\Fluent"   # folder containing JPEGs
output_video = r"Mach_Test.mp4"
fps = 120
# -----------------------------

# CROP SETTINGS (EDIT THESE)
# Example: frame[y_start:y_end, x_start:x_end]
y_start = 260
y_end   = 500
x_start = 220
x_end   = 800
# ---------------------------------------

# Collect and sort JPEGs
images = [img for img in os.listdir(image_folder)
          if img.lower().endswith((".jpg", ".jpeg"))]
images = natsort.natsorted(images)

if len(images) == 0:
    raise RuntimeError("No JPG/JPEG images found.")

# Read first frame to determine video resolution **after crop**
sample_img = cv2.imread(os.path.join(image_folder, images[0]))
cropped_sample = sample_img[y_start:y_end, x_start:x_end]
height, width, _ = cropped_sample.shape

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for name in images:
    frame_path = os.path.join(image_folder, name)
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Skipping unreadable frame: {name}")
        continue

    # Apply crop
    cropped = frame[y_start:y_end, x_start:x_end]
    video.write(cropped)

video.release()
print("Cropped video saved to:", output_video)