import whisperx
import torch
import yaml
import json
import os
import gc

config_file =  os.path.join('config', 'config.yaml')
device = "cuda"
batch_size = 16  # reduce if low on GPU mem
compute_type = "float16"  # change to "int8" if low on GPU mem (may reduce accuracy)


print(torch.cuda.is_available())

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

# Example usage
# write_config(config_file, "en", "large-v2", False, False, "your_actual_hf_token")

config = read_config(config_file)
whisper_language = config.get('language')  # Access the language setting
whisper_model = config.get('model')
whisper_folder = config.get('one_folder')
whisper_diarize = config.get('diarize')
whisper_hf_token = config.get('hf_token')

audio_file = config.get('audio_file')
model_dir = config.get('model_dir')

# 1. Transcribe with original whisper (batched)

# Save model to local path (optional)

model = whisperx.load_model(whisper_model, device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"])  # before alignment

# Delete model if low on GPU resources
# gc.collect()
# torch.cuda.empty_cache()
# del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=whisper_language, device=device)

result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"])  # after alignment

# Delete model if low on GPU resources
# gc.collect()
# torch.cuda.empty_cache()
# del model_a

if whisper_diarize:
    try:
        # 3. Assign speaker labels
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=whisper_hf_token, device=device)

        # Add min/max number of speakers if known
        diarize_segments = diarize_model(audio)
        # diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

        result = whisperx.assign_word_speakers(diarize_segments, result)

        diarize_data = diarize_segments

        print(diarize_segments)

        segment_data = result["segments"]  # segments are now assigned speaker IDs
        # Save to JSON file
        with open('segment_data.json', 'w') as json_file:
            json.dump(segment_data, json_file, indent=4)

        print(result["segments"])  # segments are now assigned speaker IDs

    except Exception as e:
        print(f"Diarization Failed: {e}")
else:
    print("Diarize not enabled")