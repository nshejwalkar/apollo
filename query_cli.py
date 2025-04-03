import sys
import time
from app import handler, generate

def main():
  if len(sys.argv) != 2:
    print("Usage: python query_cli.py <video_file>")
    sys.exit(1)

  video_file = sys.argv[1]

  message = []     
  chatbot = []     
  image = None     
  video = video_file

  temperature = 0.4
  top_p = 0.7
  max_output_tokens = 256

  print("Video loaded. You can now ask questions about the video.")
  print("Type 'exit' or 'quit' to end the session.\n")

  while True:
    # Get user input from the command line
    question = input("Enter your question: ")
    st_time = time.time()
    if question.lower() in ['exit', 'quit']:
        print("Exiting session.")
        break

    _, _, message, chatbot, _ = generate(
        image, video, message, chatbot, question, temperature, top_p, max_output_tokens
    )

    end_time = time.time()

    print("Assistant:", chatbot[-1][1])
    print("Time taken %.2f seconds" % (end_time - st_time))
    print("-" * 50)

if __name__ == '__main__':
  main()
