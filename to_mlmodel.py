#!/usr/bin/env python3

# A script for converting Spleeter (or at least part of it) to Core ML.

# Useful resources:
# - https://github.com/deezer/spleeter/issues/210
# - https://github.com/deezer/spleeter/issues/155
# - https://twitter.com/ExtractorVocal/status/1342643493227773952

import coremltools as ct
import numpy as np
import librosa
import soundfile
import torch

from pathlib import Path
from spleeter_pytorch.estimator import Estimator

ROOT = Path(__file__).resolve().parent

def main():
    samplerate = 44100
    estimator = Estimator(2, ROOT / 'checkpoints' / '2stems' / 'model')
    estimator.eval()

    # Load wav audio
    wav, _ = librosa.load(ROOT / 'audio_example.mp3', mono=False, res_type='kaiser_fast', sr=samplerate)
    wav = torch.Tensor(wav)

    # Reproduce the STFT step (which we cannot convert to Core ML, unfortunately)
    _, stft_mag = estimator.compute_stft(wav)

    print('==> Tracing')
    traced_model = torch.jit.trace(estimator.separator, stft_mag)
    out = traced_model(stft_mag)

    print('==> Converting') # TODO: Dynamic input size?
    mlmodel = ct.convert(traced_model, convert_to='mlprogram', inputs=[ct.TensorType(shape=stft_mag.shape)])
    print(mlmodel)

if __name__ == '__main__':
    main()
