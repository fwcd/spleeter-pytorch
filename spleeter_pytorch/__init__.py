import argparse
import numpy as np
import librosa
import soundfile
import torch

from pathlib import Path

from spleeter_pytorch.estimator import Estimator

ROOT = Path(__file__).resolve().parent.parent

def main():
    parser = argparse.ArgumentParser(description='Separate stems from an audio file.')
    parser.add_argument('-m', '--model', type=Path, default=ROOT / 'checkpoints' / '2stems' / 'model', help='The path to the model to use.')
    parser.add_argument('-o', '--output', type=Path, default=ROOT / 'output', help='The path to the output directory.')
    parser.add_argument('input', type=Path, help='The path to the input file to process')

    args = parser.parse_args()
    samplerate = 44100
    es = Estimator(2, args.model)
    es.eval()

    # load wav audio
    wav, _ = librosa.load(args.input, mono=False, res_type='kaiser_fast',sr=samplerate)
    wav = torch.Tensor(wav)

    # normalize audio
    # wav_torch = wav / (wav.max() + 1e-8)

    wavs = es.separate(wav)
    for i in range(len(wavs)):
        fname = args.output / f'out_{i}.wav'
        print(f'Writing {fname}')
        soundfile.write(fname, wavs[i].cpu().detach().numpy().T, samplerate, "PCM_16")
        # write_wav(fname, np.asfortranarray(wavs[i].squeeze().numpy()), sr)
