import whisperx
import torch
import yaml
import json
import os
import subprocess
import re
import unicodedata
from pydub import AudioSegment
from tkinter import filedialog, Tk

config_file = os.path.join('config', 'config.yaml')
model_dir = r'models\whisper_models'
output_dir = r'data\output'

device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)
save_audio_ext = '.wav'

print(torch.cuda.is_available())

# Default configuration values
whisper_language = 'en'
whisper_model = "large-v2"
whisper_folder = False
whisper_diarize = False
whisper_hf_token = 'HF_Token'

def read_config(config_file):
    """
    Reads the configuration from a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print("Config file not found.")
        return {}

def write_config(config_file, language, model, one_folder, diarize, hf_token):
    """
    Writes the configuration to a YAML file.

    Args:
        config_file (str): Path to the YAML configuration file.
        language (str): Language code.
        model (str): Model name.
        one_folder (bool): One folder flag.
        diarize (bool): Diarize flag.
        hf_token (str): Hugging Face token.
    """
    config = {
        'language': language,
        'model': model,
        'one_folder': one_folder,
        'diarize': diarize,
        'hf_token': hf_token
    }
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def load_config():
    global whisper_language, whisper_model, whisper_folder, whisper_diarize, whisper_hf_token
    config = read_config(config_file)
    whisper_language = config.get('language', whisper_language)
    whisper_model = config.get('model', whisper_model)
    whisper_folder = config.get('one_folder', whisper_folder)
    whisper_diarize = config.get('diarize', whisper_diarize)
    whisper_hf_token = config.get('hf_token', whisper_hf_token)

def sanitize_filename(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    # Remove diacritics and normalize Unicode characters
    normalized = unicodedata.normalize('NFKD', filename)
    sanitized = ''.join(c for c in normalized if not unicodedata.combining(c))
    # Regular Expression to match invalid characters
    invalid_chars_pattern = r'[<>:"/\\|?*]'
    # Replace invalid characters with an underscore
    return re.sub(invalid_chars_pattern, '_', sanitized)


def split_audio(json_filename, audio_file_path, new_output_dir):
        # Load the JSON file
    with open(json_filename, 'r') as file:
        data = json.load(file)

    # Load the WAV file
    audio = AudioSegment.from_wav(audio_file_path)

    # Create directories for each speaker
    speakers = set(item['speaker'] for item in data)
    for speaker in speakers:
        speaker_file_path = os.path.join(new_output_dir, speaker)
        os.makedirs(speaker_file_path, exist_ok=True)

    # Process each sentence
    for index, item in enumerate(data):
        start_time = item['start'] * 1000  # Convert to milliseconds
        end_time = item['end'] * 1000  # Convert to milliseconds
        text = item['text']
        speaker = item['speaker']

        # Extract the segment
        segment = audio[start_time:end_time]

        # Save the segment to the corresponding speaker's directory
        speaker_file_path = os.path.join(new_output_dir, speaker)
        segment_path = os.path.join(speaker_file_path, f"{start_time}_{end_time}.wav")
        segment.export(segment_path, format="wav")

        # Convert start and end times to SRT format (hh:mm:ss,mmm)
        start_srt = f"{int(start_time // 3600000):02}:{int((start_time % 3600000) // 60000):02}:{int((start_time % 60000) // 1000):02},{int(start_time % 1000):03}"
        end_srt = f"{int(end_time // 3600000):02}:{int((end_time % 3600000) // 60000):02}:{int((end_time % 60000) // 1000):02},{int(end_time % 1000):03}"

        # Save the text to a subtitle file in the corresponding speaker's directory
        subtitle_path = os.path.join(speaker_file_path, f"{start_time}_{end_time}.srt")
        with open(subtitle_path, 'w') as subtitle_file:
            subtitle_file.write(f"{index + 1}\n")
            subtitle_file.write(f"{start_srt} --> {end_srt}\n")
            subtitle_file.write(f"{text}\n")


    print("Processing complete.")

def run_whisperx(audio_file_path, new_output_dir, audio_filename):
    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model(whisper_model, device, compute_type=compute_type, download_root=model_dir)
    

    audio = whisperx.load_audio(audio_file_path)
    result = model.transcribe(audio, batch_size=batch_size)
    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=whisper_language, device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    # Create the path for the new .json file
    json_filename = os.path.join(new_output_dir, f"{audio_filename}.json")

    if whisper_diarize:
        try:
            # 3. Assign speaker labels
            diarize_model = whisperx.DiarizationPipeline(use_auth_token=whisper_hf_token, device=device)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(diarize_segments)
            segment_data = result["segments"]  # segments are now assigned speaker IDs
        except Exception as e:
            print(f"Diarization Failed: {e}")
    else:
        print("Diarize not enabled")
    # Open the file and write the data in JSON format
    print(result["segments"])

    with open(json_filename, 'w') as json_file:
        json.dump(result["segments"], json_file, indent=4)

    split_audio(json_filename, audio_file_path, new_output_dir)


def select_audio_files():
    # Initialize tkinter root window (it won't show up)
    root = Tk()
    root.withdraw()  # Hide the main tkinter window
    # Ask user to select a folder
    folder_path = filedialog.askdirectory(title="Select Folder Containing Audio Files")
    # Check if folder was selected
    if folder_path:
        # List of audio file extensions you want to include
        audio_extensions = ('.mp3', '.wav', '.flac', '.aac', '.ogg')
        # Get all files in the selected folder
        audio_files = [f for f in os.listdir(folder_path) if f.lower().endswith(audio_extensions)]
        print(f"Selected folder: {folder_path}")
        print("Audio files in folder:")
        for audio_file in audio_files:
            print(audio_file)
        return folder_path, audio_files
    else:
        print("No folder selected or no audio files found.")
        return None, None

def process_audio_files(input_folder):

    for audio_file in os.listdir(input_folder):

        audio_file_path = os.path.join(input_folder, audio_file)

        audio_filename = sanitize_filename(audio_file_path)
        new_output_dir = os.path.join(output_dir, audio_filename)

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(new_output_dir, exist_ok=True)


        if not os.path.isfile(audio_file_path):
            continue
        if not audio_file.endswith(save_audio_ext):
            wav_file_path = os.path.join(new_output_dir, f"{audio_filename}{save_audio_ext}")
            try:
                subprocess.run(['ffmpeg', '-i', audio_file_path, wav_file_path], check=True)
                audio_file_path = wav_file_path
            except subprocess.CalledProcessError as e:
                print(f"Error: {e.output}. Couldn't convert {audio_file} to {save_audio_ext} format.")
                continue
        run_whisperx(audio_file_path, new_output_dir, audio_filename)


def main():
    input_folder, audio_files = select_audio_files()
    load_config()
    process_audio_files(input_folder)

if __name__ == "__main__":
    main()
