#!/usr/bin/env python3

import coremltools as ct
import numpy as np
import librosa
import soundfile
import torch

from pathlib import Path
from spleeter.estimator import Estimator

ROOT = Path(__file__).resolve().parent

def main():
    sr = 44100
    es = Estimator(2, ROOT / 'checkpoints' / '2stems' / 'model')
    es.eval()

    # load wav audio
    wav, _ = librosa.load(ROOT / 'audio_example.mp3', mono=False, res_type='kaiser_fast',sr=sr)
    wav = torch.Tensor(wav)

    # normalize audio
    # wav_torch = wav / (wav.max() + 1e-8)

    print('==> Tracing')
    traced_model = torch.jit.trace(es, wav)
    print('==> Inferring')
    out = traced_model(wav)
    print('==> Converting')
    mlmodel = ct.convert(traced_model, convert_to='mlprogram', inputs=[ct.TensorType(shape=wav.shape)])
    print(mlmodel)

    wavs = es.separate(wav)
    for i in range(len(wavs)):
        fname = 'output/out_{}.wav'.format(i)
        print('Writing ',fname)
        soundfile.write(fname, wavs[i].cpu().detach().numpy().T, sr, "PCM_16")
        # write_wav(fname, np.asfortranarray(wavs[i].squeeze().numpy()), sr)

if __name__ == '__main__':
    main()
