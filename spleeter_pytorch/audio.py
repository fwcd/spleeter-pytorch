from pathlib import Path

import subprocess
import tempfile
import torch
import torchaudio

def read_wav(path: Path) -> tuple[torch.Tensor, int]:
    '''
    Reads a waveform from a WAV audio file.
    '''

    return torchaudio.load(path)

def write_wav(path: Path, wav: torch.Tensor, samplerate: int):
    '''
    Writes a waveform to a WAV audio file.
    '''

    torchaudio.save(path, wav, samplerate)

def read_audio(path: Path) -> tuple[torch.Tensor, int]:
    '''
    Reads an audio file in an arbitrary format supported
    by ffmpeg to a waveform and a samplerate.
    '''

    if path.name.endswith('.wav'):
        return read_wav(path)
    
    with tempfile.TemporaryDirectory(prefix='spleeter-pytorch-conversion-') as tmpdir:
        wav_path = Path(tmpdir) / 'out.wav'

        # TODO: Make sure that ffmpeg is on the user's PATH and emit an error message otherwise
        subprocess.run(
            ['ffmpeg', '-i', str(path), wav_path],
            check=True
        )

        return read_wav(wav_path)
