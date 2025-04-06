import sys
import time
import psutil
import torch
import shutil
import argparse
from app import handler, generate

def get_ram_usage():
  process = psutil.Process()
  mem_info = process.memory_info()
  return mem_info.rss / (1024 ** 2)  # MB

def get_vram_usage():
  if torch.cuda.is_available():
      return torch.cuda.memory_allocated() / (1024 ** 2)  # MB
  return 0

def get_vram_reserved():
  if torch.cuda.is_available():
      return torch.cuda.memory_reserved() / (1024 ** 2)  # MB
  return 0

def get_vram_peak():
  if torch.cuda.is_available():
      return torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
  return 0

def get_disk_usage(path="."):
  usage = shutil.disk_usage(path)
  return {
      "total": usage.total / (1024 ** 3),  # GB
      "used": usage.used / (1024 ** 3),
      "free": usage.free / (1024 ** 3),
  }

def main():
  parser = argparse.ArgumentParser(description="Query Apollo video model.")
  parser.add_argument("video_file", help="Path to the input video file")
  parser.add_argument("question", nargs="?", help="Question to ask about the video")
  parser.add_argument("--preset", choices=["summary_title"], help="Use a preset question format")
  args = parser.parse_args()
  
  video_file = args.video_file
  question = args.question

  if args.preset == "summary_title":
      question = (
          "Analyze the video and return a short, catchy title followed by a concise summary. "
          "Format the response as:\n"
          "Title: <one-sentence title>\n"
          "Summary: <two to three sentence description of what is happening in the video, "
          "including important visual details>"
      )

  if not question:
      print("Error: You must provide either a question or a preset.")
      sys.exit(1)

  message = []
  chatbot = []
  image = None
  video = video_file

  temperature = 0.4
  top_p = 0.7
  max_output_tokens = 256

  print(f"Running inference on video: {video_file}")
  print(f"Question: {question}\n")

  wall_start = time.time()
  ram_before = get_ram_usage()
  vram_before = get_vram_usage()
  disk_before = get_disk_usage()

  torch.cuda.reset_peak_memory_stats()
  torch.cuda.empty_cache()

  infer_start = time.time()
  _, _, message, chatbot, _ = generate(
      image, video, message, chatbot, question, temperature, top_p, max_output_tokens
  )
  infer_end = time.time()

  wall_end = time.time()
  ram_after = get_ram_usage()
  vram_after = get_vram_usage()
  vram_reserved = get_vram_reserved()
  vram_peak = get_vram_peak()
  disk_after = get_disk_usage()

  print("Assistant:", chatbot[-1][1])
  print("\n📊 Stats:")
  print(f"- Wall time:       {wall_end - wall_start:.2f} sec")
  print(f"- Inference time:  {infer_end - infer_start:.2f} sec")
  print(f"- RAM used:        {ram_after - ram_before:.2f} MB")
  print(f"- VRAM used:       {vram_after - vram_before:.2f} MB")
  print(f"- VRAM peak:       {vram_peak:.2f} MB")
  print(f"- VRAM reserved:   {vram_reserved:.2f} MB")
  print(f"- Disk used:       {disk_after['used'] - disk_before['used']:.2f} GB")
  print(f"- Disk free:       {disk_after['free']:.2f} GB remaining")

if __name__ == '__main__':
  main()
