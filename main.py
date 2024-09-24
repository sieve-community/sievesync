import sieve
from typing import Literal
import subprocess
import os
import concurrent.futures
import uuid
from utils import generate_timestamp_chunks, content_padding
from alignment import align_media, unalign_media

# Import Sieve functions
musetalk = sieve.function.get("sieve/musetalk:f2108181-bf55-43a6-bc5e-6725ba663a12")
codeformer = sieve.function.get("sieve/codeformer:bae00b7f-60de-4a2a-af53-37c2082ae709")
audio_enhance = sieve.function.get("sieve/audio_enhancement")
liveportrait = sieve.function.get("sieve/liveportrait:f5f20f7b-e4a4-41bb-8306-1d7b522ffba7")

metadata = sieve.Metadata(
    title="SieveSync",
    description="A lipsync pipeline built with MuseTalk, LivePortrait, and CodeFormer.",
    image=sieve.Image(path="lipsync.png"),
    readme=open("README.md").read()
)

@sieve.function(
    name="sievesync",
    python_packages=["opencv-python", "numpy==1.24.4", "mediapipe==0.10.11", "ffmpeg-python", "scikit-image"],
    system_packages=["ffmpeg"],
    metadata=metadata,
    python_version="3.10"
)
def lipsync(
    file: sieve.File,
    audio: sieve.File,
):
    '''
    :param file: Image or video file to lipsync.
    :param audio: Audio file to sync with the driver image/video file.
    :return: Lipsynced video file.
    '''

    print("Starting lipsync")
    # Create a temporary directory
    temp_dir = os.path.join(os.getcwd(), f"temp_{uuid.uuid4()}")
    os.makedirs(temp_dir, exist_ok=True)
    output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        audio_path = prepare_audio(audio, temp_dir)
        video_path, audio_path, image_input = prepare_video(file, audio_path, "audio", temp_dir)
        
        align_face = True
        enhance_audio = True
        align_dict = None
        if not image_input:
            try:
                video_path, source_video_path, align_face, align_dict = align_video(video_path, temp_dir)
            except Exception as e:
                print("Could not align video:", e)
                print("Continuing with original video")
                align_face = False
        else:
            if backend == "sievesync":
                backend = "musetalk"

        if align_dict:
            avg_face = align_dict['avg_face_size']
        else:
            avg_face = 0
        
        audio_duration = get_audio_duration(audio_path)
        timestamp_chunks = generate_timestamp_chunks(audio_duration, [])

        print("Enhancing audio for lipsync")
        try:
            lipsync_audio = audio_enhance.run(sieve.File(audio_path), filter_type="noise")
        except Exception as e:
            print("Error enhancing audio:", e)
            print("Continuing with original audio")
            lipsync_audio = sieve.File(path=audio_path)

        print("Running lipsync")
        try:
            if align_face:
                # parallelize the neutralization process by chunking the audio
                timestamp_chunks = generate_timestamp_chunks(get_audio_duration(lipsync_audio.path), [])
                neutral_video = neutralize_with_liveportrait(video_path, timestamp_chunks, temp_dir)
                lipsync_output = musetalk.run(sieve.File(path=neutral_video), lipsync_audio, smooth=False, downsample=False, override=15)
            else:
                lipsync_output = musetalk.run(sieve.File(path=video_path), lipsync_audio, smooth=False, downsample=False)
        except Exception as e:
            print("Error creating neutral face video using LivePortrait:", e)
            print("Continuing with original video")
            lipsync_output = musetalk.run(sieve.File(path=video_path), lipsync_audio, smooth=False, downsample=False)
        
        print("Post processing")
        blend = compute_ideal_blend(avg_face, False)
        final_output = enhance_with_codeformer(lipsync_output, timestamp_chunks, align_face, blend, image_input, temp_dir)
        
        duration = get_video_duration(final_output)
        print("Duration of the final output:", duration)
        
        if (not image_input) and align_face:
            print("Unaligning video")
            final_output = unalign_video(final_output, source_video_path, align_dict, temp_dir)
        
        final_output_with_audio = add_audio_to_output(final_output, audio_path, duration, audio_duration, temp_dir)
        
        # Move the final output to the current working directory
        final_output_path = os.path.join(output_dir, os.path.basename(final_output_with_audio))
        os.rename(final_output_with_audio, final_output_path)
        
        return sieve.File(path=final_output_path)
    
    finally:
        # Clean up the temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

def prepare_audio(audio, temp_dir):
    if audio.path.endswith(".mp4"):
        print("Extracting audio from video file")
        temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
        convert_to_audio = f"ffmpeg -y -i '{audio.path}' -vn -acodec pcm_s16le -ar 44100 -ac 2 '{temp_audio_path}'"
        os.system(convert_to_audio)
        return temp_audio_path
    return audio.path

def prepare_video(file, audio_path, cut_by, temp_dir):
    if file.path.endswith((".jpg", ".jpeg", ".png")):
        video_path, audio_path = content_padding(file, sieve.File(audio_path), cut_by, temp_dir)
        return video_path, audio_path, False
    else:
        video_path, audio_path = content_padding(file, sieve.File(audio_path), cut_by, temp_dir)
        return video_path, audio_path, False

def align_video(video_path, temp_dir):
    try:
        aligned_path = os.path.join(temp_dir, f"aligned_{uuid.uuid4()}.mp4")
        align_dict = align_media(video_path, aligned_path)
        return aligned_path, video_path, True, align_dict
    except Exception as e:
        print("Error aligning video:", e)
        print("Continuing with original video")
        return video_path, video_path, False, None

def get_audio_duration(audio_path):
    cmd = ['ffprobe', '-i', audio_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return float(result.stdout)

def compute_ideal_blend(avg_face: float, downsample: bool) -> float:
    base_max = 0.75
    base_min = 0.6
    
    if avg_face <= 2:
        return base_max
    elif avg_face >= 10:
        return base_min
    else:
        # Linear interpolation
        slope = (base_min - base_max) / (10 - 2)
        return base_max + slope * (avg_face - 2)

def process_neutralize_chunk(video_path, chunk):
    return liveportrait.push(sieve.File(path=video_path), sieve.File(path="lipsync.png"), input_lip_ratio=0, neutralize_first_frame=False, source_max_dim=1920, flag_lip_retargeting=True, start_time_source=chunk[0], end_time_source=chunk[1])

def neutralize_with_liveportrait(video_path, timestamp_chunks, temp_dir):
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_neutralize_chunk, video_path, chunk) for chunk in timestamp_chunks]
            processed_outputs = [future.result() for future in futures]
        
        return combine_outputs(processed_outputs, temp_dir)
    except Exception as e:
        print("Error neutralizing video using LivePortrait:", e)
        print("Continuing with original video")
        return video_path

def enhance_with_codeformer(lipsync_output, timestamp_chunks, align_face, blend, image_input, temp_dir):
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_chunk, lipsync_output.path, chunk, blend, align_face, image_input) for chunk in timestamp_chunks]
            processed_outputs = [future.result() for future in futures]
        
        return combine_outputs(processed_outputs, temp_dir)
    except Exception as e:
        print("Error enhancing output using CodeFormer:", e)
        return lipsync_output.path

def process_chunk(lipsync_output_path, chunk, blend, align_face, image_input):
    if image_input:
        return codeformer.push(sieve.File(path=lipsync_output_path), fidelity_weight=1, upscale=1, blend_ratio=blend, start_time=chunk[0], end_time=chunk[1], has_aligned=False)
    else:
        return codeformer.push(sieve.File(path=lipsync_output_path), fidelity_weight=1, upscale=1, blend_ratio=blend, start_time=chunk[0], end_time=chunk[1], has_aligned=align_face)

def combine_outputs(processed_outputs, temp_dir):
    output_files = [output.result().path for output in processed_outputs]
    output_list = os.path.join(temp_dir, "concat_list.txt")
    print("Combining processed chunks")
    with open(output_list, "w") as f:
        for file in output_files:
            f.write(f"file '{file}'\n")
    
    final_output = os.path.join(temp_dir, f"combined_output_{uuid.uuid4()}.mp4")
    ffmpeg_cmd = f"ffmpeg -f concat -safe 0 -i {output_list} -c copy {final_output}"
    subprocess.run(ffmpeg_cmd, shell=True, check=True)
    os.remove(output_list)
    return final_output

def get_video_duration(video_path):
    return float(os.popen(f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {video_path}").read())

def unalign_video(final_output, source_video_path, align_dict, temp_dir):
    try:
        unaligned_path = os.path.join(temp_dir, f"unaligned_{uuid.uuid4()}.mp4")
        if align_dict is None:
            print("No alignment data available. Skipping unalignment.")
            return final_output
        unalign_media(final_output, source_video_path, align_dict, unaligned_path)
        return unaligned_path
    except Exception as e:
        print("Error unaligning video:", e)
        print("Continuing with aligned video")
        return final_output

def add_audio_to_output(final_output, audio_path, video_duration, audio_duration, temp_dir):
    final_output_with_audio = os.path.join(temp_dir, f"final_output_with_audio_{uuid.uuid4()}.mp4")
    final_output_escaped = final_output.replace("'", "'\\''")
    audio_path_escaped = audio_path.replace("'", "'\\''")

    print("Audio duration:", audio_duration)
    print("Video duration:", video_duration)

    if audio_duration > video_duration:
        # Speed up audio to match video duration
        speed_factor = video_duration / audio_duration
        ffmpeg_cmd = (
            f"ffmpeg -i '{final_output_escaped}' -i '{audio_path_escaped}' "
            f"-filter_complex '[1:a]atempo={speed_factor}[a]' "
            f"-c:v libx264 -preset ultrafast -crf 23 -c:a aac -b:a 128k "
            f"-map 0:v:0 -map '[a]' {final_output_with_audio}"
        )
    else:
        # Use original audio
        ffmpeg_cmd = (
            f"ffmpeg -i '{final_output_escaped}' -i '{audio_path_escaped}' "
            f"-c:v libx264 -preset ultrafast -crf 23 -c:a aac -b:a 128k "
            f"-map 0:v:0 -map 1:a:0 {final_output_with_audio}"
        )
    
    subprocess.run(ffmpeg_cmd, shell=True, check=True)
    return final_output_with_audio

if __name__ == "__main__":
    output = lipsync(
        sieve.File("elon-main.mp4"),
        sieve.File("elon-spanish.wav")
    )
    print("output video saved to", output.path)