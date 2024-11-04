from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# Function to extract multiple clips
def extract_clips(input_file, output_files, clip_times):
    for i, (start_time, end_time) in enumerate(clip_times):
        try:
            # Extract the clip and save it to the corresponding output file
            ffmpeg_extract_subclip(input_file, start_time, end_time, targetname=output_files[i])
            print(f"Clip {i+1} saved as {output_files[i]}")
        except Exception as e:
            print(f"Error extracting clip {i+1}: {e}")

# Example usage
input_mp4 = "C:/Users/Dejan/Downloads/videoplayback.mp4"
output_files = [
    "clip1.mp4",
    "clip2.mp4",
    "clip3.mp4",
    "clip4.mp4"
]

# List of (start, end) times in seconds for each clip
clip_times = [
    (32, 42),  # Clip 1 from 10s to 20s
    (67, 77),  # Clip 2 from 30s to 40s
    (150, 160),  # Clip 3 from 50s to 60s
    (260, 266.5)   # Clip 4 from 70s to 80s
]

extract_clips(input_mp4, output_files, clip_times)

