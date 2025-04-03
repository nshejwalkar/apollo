import sys
import time
import os
import psutil
import torch
from app import handler, generate

# Function to calculate the total size (in bytes) of a directory.
def get_directory_size(directory):
   total_size = 0
   for dirpath, _, filenames in os.walk(directory):
      for f in filenames:
         fp = os.path.join(dirpath, f)
         if os.path.exists(fp):
               total_size += os.path.getsize(fp)
   return total_size

def print_metrics(pair_time):
   # Print query execution time
   print(f"\nTime taken for both queries: {pair_time:.2f} seconds")
   
   # Print disk usage for the model (assumed to be in "./tmp/data")
   model_dir = "./tmp/data"
   model_disk_usage = get_directory_size(model_dir) / (1024 * 1024)  # in MB
   print(f"Model disk usage: {model_disk_usage:.2f} MB")
   
   # Print RAM usage for the current process.
   process = psutil.Process(os.getpid())
   ram_usage = process.memory_info().rss / (1024 * 1024)  # in MB
   print(f"RAM usage (current process): {ram_usage:.2f} MB")
   
   # Print VRAM usage if CUDA is available.
   if torch.cuda.is_available():
      vram_usage = torch.cuda.memory_allocated() / (1024 * 1024)  # in MB
      print(f"VRAM usage: {vram_usage:.2f} MB")
   print("-" * 50)

def process_pair(prompt1, prompt2, message, chatbot, image, video,
               temperature=0.4, top_p=0.7, max_output_tokens=256):
   # Process the first prompt
   _, _, message, chatbot, _ = generate(
      image, video, message, chatbot, prompt1, temperature, top_p, max_output_tokens
   )
   print("\nAssistant reply to prompt1:")
   print(chatbot[-1][1])
   
   # Process the second prompt
   _, _, message, chatbot, _ = generate(
      image, video, message, chatbot, prompt2, temperature, top_p, max_output_tokens
   )
   print("\nAssistant reply to prompt2:")
   print(chatbot[-1][1])
   
   return message, chatbot

def main():
   if len(sys.argv) != 4:
      print("Usage: python query_cli.py <video_file> <prompt1> <prompt2>")
      sys.exit(1)

   video_file = sys.argv[1]
   prompt1 = sys.argv[2]
   prompt2 = sys.argv[3]

   # Initialize conversation state once.
   message = []     # conversation message history
   chatbot = []     # chat log (list of [user_message, assistant_reply])
   image = None     # we use video only.
   video = video_file

   temperature = 0.4
   top_p = 0.7
   max_output_tokens = 256

   # Process the two prompts from the command line.
   print("Processing initial prompts...")
   start_time = time.time()
   message, chatbot = process_pair(prompt1, prompt2, message, chatbot, image, video,
                                 temperature, top_p, max_output_tokens)
   end_time = time.time()

   # Print the metrics after the pair of queries.
   print_metrics(end_time - start_time)

   # Now clear the conversation history and allow new pairs to be entered interactively.
   while True:
      print("\n--- Conversation history cleared. Enter new prompt pair ---")
      # Clear the conversation state.
      message = []
      chatbot = []
      new_prompt1 = input("Enter prompt1 (or type 'exit' to quit): ")
      if new_prompt1.lower() in ["exit", "quit"]:
         break
      new_prompt2 = input("Enter prompt2 (or type 'exit' to quit): ")
      if new_prompt2.lower() in ["exit", "quit"]:
         break

      start_time = time.time()
      message, chatbot = process_pair(new_prompt1, new_prompt2, message, chatbot, image, video,
                                       temperature, top_p, max_output_tokens)
      end_time = time.time()
      print_metrics(end_time - start_time)

if __name__ == '__main__':
   main()
