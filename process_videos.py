import os
import subprocess
import cv2

INPUT_DIR = "videos_to_process"
OUTPUT_DIR = "videos"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_video_duration_seconds(video_path):
   cap = cv2.VideoCapture(video_path)
   if not cap.isOpened():
      return None
   fps = cap.get(cv2.CAP_PROP_FPS)
   frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
   cap.release()
   if fps == 0:
      return None
   return round(frame_count / fps)

def process_video(input_path, output_path):
   cmd = [
      "ffmpeg",
      "-i", input_path,
      "-vf", "fps=4,scale=384:trunc(ow/a/2)*2",
      "-c:v", "libx264",
      "-crf", "25",
      "-preset", "fast",
      output_path
   ]
   subprocess.run(cmd, check=True)

def main():
   for filename in os.listdir(INPUT_DIR):
      if not filename.lower().endswith((".mp4", ".mov", ".mkv", ".webm")):
         continue
      
      input_path = os.path.join(INPUT_DIR, filename)
      duration = get_video_duration_seconds(input_path)
      if duration is None:
         print(f"❌ Could not determine duration for {filename}")
         continue

      output_filename = f"{duration}_tiny.mp4"
      output_path = os.path.join(OUTPUT_DIR, output_filename)

      print(f"🔄 Processing {filename} -> {output_filename}")
      try:
         process_video(input_path, output_path)
         print(f"✅ Saved to {output_path}")
      except subprocess.CalledProcessError:
         print(f"❌ FFmpeg failed on {filename}")

if __name__ == "__main__":
   main()
