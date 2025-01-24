# audio-splitter-diarization-X

Currently WIP

Using whisperx to transcribe audio with speaker diarization. Then will seperate audio files into segments based on the speaker.



WHISPER X - BSD 2-Clause License

Setup ⚙️
Tested for PyTorch 2.2.0 Python 3.10 (use other versions at your own risk!)
GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the CTranslate2 documentation.

1. Create Python3.10 environment

py -3.10 -m venv venv
venv/scripts/activate

2. Install PyTorch, e.g. for Linux and Windows CUDA12.1: #Updated Torch version for compatability

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 -f https://download.pytorch.org/whl/torch_stable.html


See other methods here.

3. Install WhisperX 3.3.1

pip install whisperx==3.3.1

4. If errors, try reinstall pytorch:


pip uninstall torch torchvision torchaudio

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121 -f https://download.pytorch.org/whl/torch_stable.html


