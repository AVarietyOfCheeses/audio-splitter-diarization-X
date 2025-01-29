import sys
import os
import yaml
import json
import torch
import whisperx
import unicodedata
import re
from pydub import AudioSegment
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
                               QCheckBox, QPushButton, QFileDialog, QMessageBox, QProgressBar, QSpinBox)
from PySide6.QtCore import Qt, QThread, Signal
import subprocess
import time

# Enable TF32 for better performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class Worker(QThread):
    progress_update = Signal(int)
    gpu_usage_update = Signal(str)

    def __init__(self, input_folder, output_dir, device, batch_size, compute_type, model_dir, whisper_model, whisper_language, whisper_diarize, whisper_hf_token, min_speakers, max_speakers):
        super().__init__()
        self.input_folder = input_folder
        self.output_dir = output_dir
        self.device = device
        self.batch_size = batch_size
        self.compute_type = compute_type
        self.model_dir = model_dir
        self.whisper_model = whisper_model
        self.whisper_language = whisper_language
        self.whisper_diarize = whisper_diarize
        self.whisper_hf_token = whisper_hf_token
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

    def run(self):
        total_files = len(os.listdir(self.input_folder))
        processed_files = 0

        for audio_file in os.listdir(self.input_folder):
            audio_file_path = os.path.join(self.input_folder, audio_file)
            audio_filename = self.sanitize_filename(audio_file_path)
            new_output_dir = os.path.join(self.output_dir, audio_filename)
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(new_output_dir, exist_ok=True)

            if not os.path.isfile(audio_file_path):
                continue

            audio_ext = os.path.splitext(audio_file)[1][1:]
            self.run_whisperx(audio_file_path, new_output_dir, audio_filename, audio_ext)

            processed_files += 1
            progress = int((processed_files / total_files) * 100)
            self.progress_update.emit(progress)

            # Monitor GPU usage
            gpu_usage = self.get_gpu_usage()
            self.gpu_usage_update.emit(gpu_usage)

    def sanitize_filename(self, filepath):
        filename = os.path.splitext(os.path.basename(filepath))[0]
        normalized = unicodedata.normalize('NFKD', filename)
        sanitized = ''.join(c for c in normalized if not unicodedata.combining(c))
        invalid_chars_pattern = r'[<>:"/\\|?*]'
        return re.sub(invalid_chars_pattern, '_', sanitized)

    def run_whisperx(self, audio_file_path, new_output_dir, audio_filename, audio_ext):
        model = whisperx.load_model(self.whisper_model, self.device, compute_type=self.compute_type, download_root=self.model_dir)
        audio = whisperx.load_audio(audio_file_path)
        result = model.transcribe(audio, batch_size=self.batch_size)
        model_a, metadata = whisperx.load_align_model(language_code=self.whisper_language, device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)
        json_filename = os.path.join(new_output_dir, f"{audio_filename}.json")

        if self.whisper_diarize:
            try:
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=self.whisper_hf_token, device=self.device)
                diarize_segments = diarize_model(audio, min_speakers=self.min_speakers, max_speakers=self.max_speakers)
                result = whisperx.assign_word_speakers(diarize_segments, result)

            except Exception as e:
                print(f"Diarization Failed: {e}")
        else:
            print("Diarize not enabled")

        with open(json_filename, 'w') as json_file:
            json.dump(result["segments"], json_file, indent=4)

        self.split_audio(json_filename, audio_file_path, new_output_dir, audio_ext)

    def pad_decimal(self, number, decimals=3):
        # Add a small value to ensure there's at least one decimal digit
        return f"{number + 1e-10:.{decimals}f}"

    def split_audio(self, json_filename, audio_file_path, new_output_dir, audio_ext):
        with open(json_filename, 'r', encoding='utf-8') as file:
            data = json.load(file)

        audio = AudioSegment.from_file(audio_file_path, format=audio_ext)
        speakers = set(item.get('speaker', 'UNKNOWN') for item in data)
        for speaker in speakers:
            speaker_file_path = os.path.join(new_output_dir, speaker)
            os.makedirs(speaker_file_path, exist_ok=True)

        for index, item in enumerate(data):
            # Get start and end times, then pad and convert to float
            start_time = float(self.pad_decimal(item['start']))
            end_time = float(self.pad_decimal(item['end']))

            # Convert to milliseconds
            start_time_ms = int(start_time * 1000)
            end_time_ms = int(end_time * 1000)

            text = item['text']
            speaker = item.get('speaker', 'UNKNOWN')
            segment = audio[start_time_ms:end_time_ms]
            speaker_file_path = os.path.join(new_output_dir, speaker)

            segment_path = os.path.join(speaker_file_path, f"{start_time}_{end_time}.{audio_ext}")
            segment.export(segment_path, format=audio_ext)

            start_srt = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02},{int((start_time * 1000) % 1000):03}"
            end_srt = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02},{int((end_time * 1000) % 1000):03}"

            subtitle_path = os.path.join(speaker_file_path, f"{start_time}_{end_time}.srt")
            with open(subtitle_path, 'w', encoding='utf-8') as subtitle_file:
                subtitle_file.write(f"{index + 1}\n")
                subtitle_file.write(f"{start_srt} --> {end_srt}\n")
                subtitle_file.write(f"{text}\n")

    def get_gpu_usage(self):
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,nounits,noheader'], stdout=subprocess.PIPE)
            usage = result.stdout.decode('utf-8').strip()
            return f"GPU Usage: {usage}%"
        except Exception as e:
            return f"GPU Usage: Error ({e})"

class WhisperXApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WhisperX GUI")
        self.setGeometry(100, 100, 800, 600)

        self.config_file = os.path.join('config', 'config.yaml')
        self.model_dir = r'models\whisper_models'
        self.output_dir = r'data\output'

        self.device = "cuda"
        self.batch_size = 16
        self.compute_type = "float16"

        self.whisper_language = 'en'
        self.whisper_model = "large-v2"
        self.whisper_folder = False
        self.whisper_diarize = False
        self.whisper_hf_token = 'HF_Token'
        self.min_speakers = 1
        self.max_speakers = 10

        self.initUI()
        self.load_config()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Language selection
        language_layout = QHBoxLayout()
        language_label = QLabel("Language:")
        self.language_combo = QComboBox()
        self.language_combo.addItems(["en", "ja"])
        self.language_combo.currentTextChanged.connect(self.save_config)
        language_layout.addWidget(language_label)
        language_layout.addWidget(self.language_combo)
        layout.addLayout(language_layout)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(["tiny", "base", "small", "medium", "large-v2", "large-v3"])
        self.model_combo.currentTextChanged.connect(self.save_config)
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)

        # Diarize checkbox
        self.diarize_check = QCheckBox("Enable diarization")
        self.diarize_check.stateChanged.connect(self.save_config)
        layout.addWidget(self.diarize_check)

        # Min and Max speakers selection
        speakers_layout = QHBoxLayout()
        min_speakers_label = QLabel("Min Speakers:")
        self.min_speakers_spin = QSpinBox()
        self.min_speakers_spin.setRange(1, 10)
        self.min_speakers_spin.setValue(self.min_speakers)
        self.min_speakers_spin.valueChanged.connect(self.save_config)
        max_speakers_label = QLabel("Max Speakers:")
        self.max_speakers_spin = QSpinBox()
        self.max_speakers_spin.setRange(1, 10)
        self.max_speakers_spin.setValue(self.max_speakers)
        self.max_speakers_spin.valueChanged.connect(self.save_config)
        speakers_layout.addWidget(min_speakers_label)
        speakers_layout.addWidget(self.min_speakers_spin)
        speakers_layout.addWidget(max_speakers_label)
        speakers_layout.addWidget(self.max_speakers_spin)
        layout.addLayout(speakers_layout)

        # Hugging Face token
        hf_token_layout = QHBoxLayout()
        hf_token_label = QLabel("Hugging Face Token:")
        self.hf_token_edit = QLineEdit()
        self.hf_token_edit.setEchoMode(QLineEdit.Password)  # Initially mask the token
        self.hf_token_edit.textChanged.connect(self.save_config)
        self.show_token_check = QCheckBox("Show Token")
        self.show_token_check.stateChanged.connect(self.toggle_token_visibility)
        hf_token_layout.addWidget(hf_token_label)
        hf_token_layout.addWidget(self.hf_token_edit)
        hf_token_layout.addWidget(self.show_token_check)
        layout.addLayout(hf_token_layout)

        # Select audio files button
        self.select_files_button = QPushButton("Select Audio Files")
        self.select_files_button.clicked.connect(self.select_audio_files)
        layout.addWidget(self.select_files_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # GPU usage label
        self.gpu_usage_label = QLabel("GPU Usage: ")
        layout.addWidget(self.gpu_usage_label)

        # Process button
        self.process_button = QPushButton("Process")
        self.process_button.clicked.connect(self.process_audio_files)
        layout.addWidget(self.process_button)

    def load_config(self):
        config = self.read_config(self.config_file)
        self.whisper_language = config.get('language', self.whisper_language)
        self.whisper_model = config.get('model', self.whisper_model)
        self.whisper_diarize = config.get('diarize', self.whisper_diarize)
        self.whisper_hf_token = config.get('hf_token', self.whisper_hf_token)
        self.min_speakers = config.get('min_speakers', self.min_speakers)
        self.max_speakers = config.get('max_speakers', self.max_speakers)

        self.language_combo.setCurrentText(self.whisper_language)
        self.model_combo.setCurrentText(self.whisper_model)
        self.diarize_check.setChecked(self.whisper_diarize)
        self.hf_token_edit.setText(self.whisper_hf_token)
        self.min_speakers_spin.setValue(self.min_speakers)
        self.max_speakers_spin.setValue(self.max_speakers)

    def read_config(self, config_file):
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            QMessageBox.warning(self, "Warning", "Config file not found.")
            return {}

    def save_config(self):
        config = {
            'language': self.language_combo.currentText(),
            'model': self.model_combo.currentText(),
            'diarize': self.diarize_check.isChecked(),
            'hf_token': self.hf_token_edit.text(),
            'min_speakers': self.min_speakers_spin.value(),
            'max_speakers': self.max_speakers_spin.value()
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

    def select_audio_files(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder Containing Audio Files", options=options)
        if folder_path:
            self.input_folder = folder_path
            QMessageBox.information(self, "Info", f"Selected folder: {folder_path}")

    def process_audio_files(self):
        if hasattr(self, 'input_folder'):
            self.save_config()
            self.load_config()
            self.start_processing()
        else:
            QMessageBox.warning(self, "Warning", "Please select a folder containing audio files.")

    def start_processing(self):
        self.worker = Worker(self.input_folder, self.output_dir, self.device, self.batch_size, self.compute_type,
                             self.model_dir, self.whisper_model, self.whisper_language, self.whisper_diarize, self.whisper_hf_token,
                             self.min_speakers, self.max_speakers)
        self.worker.progress_update.connect(self.update_progress)
        self.worker.gpu_usage_update.connect(self.update_gpu_usage)
        self.worker.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def update_gpu_usage(self, usage):
        self.gpu_usage_label.setText(usage)

    def toggle_token_visibility(self, state):
        if state == Qt.Checked:
            self.hf_token_edit.setEchoMode(QLineEdit.Normal)
        else:
            self.hf_token_edit.setEchoMode(QLineEdit.Password)

def main():
    app = QApplication(sys.argv)
    window = WhisperXApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
