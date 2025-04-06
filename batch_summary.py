import os
import time
import psutil
import torch
import shutil
from app import generate

PRESET_PROMPT = (
   "Analyze the video and return a short, catchy title followed by a concise summary. Format the response as:\n"
   "Title: <one-sentence title>\n"
   "Summary: <two to three sentence description of what is happening in the video, "
   "including important visual details>"
)

def get_ram_usage():
   process = psutil.Process()
   return process.memory_info().rss / (1024 ** 2)  # MB

def get_vram_usage():
   return torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

def get_vram_peak():
   return torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

def run_inference(video_path, question):
   message = []
   chatbot = []
   image = None

   torch.cuda.reset_peak_memory_stats()
   torch.cuda.empty_cache()

   ram_before = get_ram_usage()
   vram_before = get_vram_usage()
   start = time.time()

   _, _, message, chatbot, _ = generate(
      image, video_path, message, chatbot,
      question, temperature=0.4, top_p=0.7, max_output_tokens=256
   )

   end = time.time()
   ram_after = get_ram_usage()
   vram_after = get_vram_usage()
   vram_peak = get_vram_peak()

   return {
      "video": video_path,
      "output": chatbot[-1][1] if chatbot else "ERROR: No output",
      "time": end - start,
      "ram_used": ram_after - ram_before,
      "vram_used": vram_after - vram_before,
      "vram_peak": vram_peak,
   }

def main():
   input_list = "video_list.txt"
   output_log = "batch_summary_log.txt"

   if not os.path.exists(input_list):
      print(f"Missing file: {input_list}")
      return

   with open(input_list, 'r') as f:
      videos = [line.strip() for line in f if line.strip()]

   with open(output_log, 'w') as log:
      for i, video in enumerate(videos):
         print(f"▶️ [{i+1}/{len(videos)}] Processing {video} ...")
         try:
               result = run_inference(video, PRESET_PROMPT)
               log.write(f"Video: {result['video']}\n")
               log.write(f"{result['output']}\n")
               log.write(f"⏱ Time: {result['time']:.2f} sec\n")
               log.write(f"🧠 RAM used: {result['ram_used']:.2f} MB\n")
               log.write(f"🎮 VRAM peak: {result['vram_peak']:.2f} MB\n")
               log.write("-" * 60 + "\n")
         except Exception as e:
               log.write(f"❌ Error processing {video}: {e}\n")
               log.write("-" * 60 + "\n")
         log.flush()

   print(f"\n✅ Finished. Results saved to {output_log}")

if __name__ == '__main__':
   main()
