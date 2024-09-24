import os
import subprocess
import re
import uuid
import sieve
from typing import Literal
 
def detect_silences_ffmpeg(audio_file_path, silence_thresh=-50, min_silence_len=0.25):
    """
    Detects silences in an audio file and returns a list of tuples with the start and end times in milliseconds.

    :param audio_file_path: Path to the audio file.
    :param silence_thresh: Silence threshold in dB. Default is -50 dB.
    :param min_silence_len: Minimum length of silence to detect in seconds. Default is 0.25 seconds.
    :return: List of tuples with start and end times of silences in milliseconds.
    """
    cmd = [
        'ffmpeg', '-i', audio_file_path, '-af',
        f'silencedetect=noise={silence_thresh}dB:d={min_silence_len}',
        '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    output = result.stderr

    silence_intervals = []
    silence_start_pattern = re.compile(r"silence_start: (\d+(\.\d+)?)")
    silence_end_pattern = re.compile(r"silence_end: (\d+(\.\d+)?) \| silence_duration: (\d+(\.\d+)?)")
    silence_start = None

    for line in output.splitlines():
        start_match = silence_start_pattern.search(line)
        if start_match:
            silence_start = float(start_match.group(1)) * 1000  # Convert to milliseconds
        end_match = silence_end_pattern.search(line)
        if end_match:
            silence_end = float(end_match.group(1)) * 1000  # Convert to milliseconds
            if silence_start is not None:
                silence_intervals.append((int(silence_start), int(silence_end)))
            silence_start = None

    return silence_intervals

def generate_timestamp_chunks(audio_duration, scenes=None):
    """
    Generates timestamp chunks from an audio file based on the provided scenes or a default chunk size.

    :param audio_duration: Duration of the audio in seconds.
    :param scenes: List of scene objects with 'start_seconds' and 'end_seconds'.
    :return: List of tuples with start and end times in milliseconds.
    """
    chunks = []
    
    if scenes:
        last_end = 0
        for elem in scenes:
            start = int(elem['start_seconds'] * 1000)  # Convert to milliseconds
            end = int(elem['end_seconds'] * 1000)
            print(f"Scene: {start/1000:.2f}s - {end/1000:.2f}s")
            if start > last_end:
                chunks.append((last_end, start))
            chunks.append((start, end))
            last_end = end
        
        # Add final chunk if necessary
        audio_duration_ms = int(audio_duration * 1000)
        if last_end < audio_duration_ms:
            chunks.append((last_end, audio_duration_ms))
    else:
        chunk_duration = 5000  # 5 seconds in milliseconds
        audio_duration_ms = int(audio_duration * 1000)
        for start in range(0, audio_duration_ms, chunk_duration):
            end = min(start + chunk_duration, audio_duration_ms)
            chunks.append((start, end))
    
    return [(start/1000, end/1000) for start, end in chunks]  # Convert back to seconds

def extend_video(input_video, output_video, target_duration):
    """
    Extends the video by playing it forward and backward until it reaches the target duration.

    :param input_video: Path to the input video file.
    :param output_video: Path to the output video file.
    :param target_duration: Desired duration of the output video in seconds.
    """
    # Get video duration
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-count_packets', '-show_entries', 'stream=duration',
        '-of', 'csv=p=0', input_video
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    video_duration = float(result.stdout.strip())
    
    # Calculate how many times we need to repeat the video
    repeat_count = int(target_duration / video_duration) + 1

    # Create filter complex string directly
    filter_complex = ""
    for i in range(repeat_count):
        if i % 2 == 0:
            filter_complex += f"[0:v]trim=duration={video_duration},setpts=PTS-STARTPTS[v{i}];"
        else:
            filter_complex += f"[0:v]trim=duration={video_duration},reverse,setpts=PTS-STARTPTS[v{i}];"
    
    filter_complex += "".join(f"[v{i}]" for i in range(repeat_count))
    filter_complex += f"concat=n={repeat_count}:v=1:a=0[outv]"

    # Use FFmpeg to create the extended video
    ffmpeg_cmd = [
        "ffmpeg", "-i", input_video, "-filter_complex", filter_complex,
        "-map", "[outv]", "-t", str(target_duration), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-y", output_video
    ]
    subprocess.run(ffmpeg_cmd, check=True)

def content_padding(video_file: sieve.File, audio_file: sieve.File, padding_type, temp_dir):
    """
    Pads the video and audio files based on the specified padding type. Padding types can
    be "audio" where the video is padded to the audio length, "video" where the audio is padded
    to the video length, or "shortest" where the shortest length of either the video or audio
    is used.

    :param video_file: Input video file.
    :param audio_file: Input audio file.
    :param padding_type: Type of padding to apply.
    :param temp_dir: Temporary directory for intermediate files.
    :param output_dir: Output directory for the final video and audio files.
    """
    print(f"Starting content padding with padding type: {padding_type}")
    # Get video and audio durations
    video_path, audio_path = video_file.path, audio_file.path
    
    # Check if video_path is an image
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    if any(video_path.lower().endswith(ext) for ext in image_extensions):
        # Get audio duration
        audio_duration_str = subprocess.run(['ffprobe', '-i', audio_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0'], stdout=subprocess.PIPE, text=True).stdout.strip()
        audio_duration = float(audio_duration_str.rstrip(','))
        
        # Convert image to video
        temp_video_path = os.path.join(temp_dir, f"image_to_video_{uuid.uuid4()}.mp4")
        subprocess.run(['ffmpeg', '-loop', '1', '-i', video_path, '-c:v', 'libx264', '-t', str(audio_duration), '-pix_fmt', 'yuv420p', '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2', temp_video_path], check=True)
        print(f"Image converted to video: {temp_video_path}")
        video_path = temp_video_path
        video_duration = audio_duration
    else:
        # Get video duration
        video_duration_str = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries', 'stream=duration', '-of', 'csv=p=0', video_path], stdout=subprocess.PIPE, text=True).stdout.strip()
        video_duration = float(video_duration_str.rstrip(','))
        
        # Get audio duration
        audio_duration_str = subprocess.run(['ffprobe', '-i', audio_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0'], stdout=subprocess.PIPE, text=True).stdout.strip()
        audio_duration = float(audio_duration_str.rstrip(','))
    
    print(f"Video duration: {video_duration} seconds\nAudio duration: {audio_duration} seconds")

    def escape_path(path):
        return path.replace("'", "'\\''")

    if padding_type == "audio":
        if video_duration < audio_duration:
            extended_video = os.path.join(temp_dir, f"truncated_video_{uuid.uuid4()}.mp4")
            extend_video(video_path, extended_video, audio_duration)
            return convert_to_25fps(extended_video, temp_dir), audio_path
        elif video_duration > audio_duration:
            truncated_video = os.path.join(temp_dir, f"truncated_video_{uuid.uuid4()}.mp4")
            subprocess.run(f"ffmpeg -i '{escape_path(video_path)}' -ss 0 -t {audio_duration} -c:v libx264 -preset ultrafast -crf 23 -c:a aac -strict experimental -r 25 -y '{escape_path(truncated_video)}'", shell=True, check=True)
            return truncated_video, audio_path
        return convert_to_25fps(video_path, temp_dir), audio_path

    elif padding_type == "video":
        if audio_duration > video_duration:
            truncated_audio = os.path.join(temp_dir, f"truncated_audio_{uuid.uuid4()}.wav")
            subprocess.run(f"ffmpeg -i '{escape_path(audio_path)}' -ss 0 -t {video_duration} -c:a aac -strict experimental -y '{escape_path(truncated_audio)}'", shell=True, check=True)
            return convert_to_25fps(video_path, temp_dir), truncated_audio
        return convert_to_25fps(video_path, temp_dir), audio_path

    elif padding_type == "shortest":
        shortest_duration = min(video_duration, audio_duration)
        if video_duration > shortest_duration:
            truncated_video = os.path.join(temp_dir, f"truncated_video_{uuid.uuid4()}.mp4")
            subprocess.run(f"ffmpeg -i '{escape_path(video_path)}' -ss 0 -t {shortest_duration} -c:v libx264 -preset ultrafast -crf 23 -c:a aac -strict experimental -r 25 -y '{escape_path(truncated_video)}'", shell=True, check=True)
            return truncated_video, audio_path
        elif audio_duration > shortest_duration:
            truncated_audio = os.path.join(temp_dir, f"truncated_audio_{uuid.uuid4()}.wav")
            subprocess.run(f"ffmpeg -i '{escape_path(audio_path)}' -ss 0 -t {shortest_duration} -c:a aac -strict experimental -y '{escape_path(truncated_audio)}'", shell=True, check=True)
            return convert_to_25fps(video_path, temp_dir), truncated_audio
        return convert_to_25fps(video_path, temp_dir), audio_path

def convert_to_25fps(video_path, output_dir):
    """
    Converts the video to 25fps.

    :param video_path: Path to the input video file.
    :return: Path to the output video file.
    """
    output_path = os.path.join(output_dir, f"25fps_{uuid.uuid4()}.mp4")
    subprocess.run(f"ffmpeg -i '{video_path}' -filter:v fps=fps=25 -c:v libx264 -preset ultrafast -crf 23 -c:a copy -y {output_path}", shell=True, check=True)
    return output_path
