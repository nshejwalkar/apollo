```shell
pip install -r requirements.txt
# FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip install flash-attn --no-build-isolation
# pip install yt-dlp # in case you want it to download yt video
python app.py
or
python query_cli2.py <video_path> <prompt1> <prompt2>
prompt1 could be “Return only the best matching category for this video from: Music, Sports & Fitness, Gaming, Cars, Travel, Food & Cooking, Comedy, Events & Parties”
prompt2 could be "Return 10 relevant keywords that describe what's happening in the video. Output only the keywords, comma-separated."
```
