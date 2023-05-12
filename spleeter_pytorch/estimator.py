import torch
import torch.nn.functional as F

from pathlib import Path
from torch import nn

from spleeter_pytorch.separator import Separator
from spleeter_pytorch.util import overlap_and_add

class Estimator(nn.Module):
    def __init__(self, num_instruments: int, checkpoint_path: Path):
        super().__init__()

        # stft config
        self.F = 1024
        self.T = 512
        self.win_length = 4096 # should be a power of two, see https://github.com/tensorflow/tensorflow/blob/6935c8f706dde1906e388b3142906c92cdcc36db/tensorflow/python/ops/signal/spectral_ops.py#L48-L49
        self.hop_length = 1024
        self.win = nn.Parameter(
            torch.hann_window(self.win_length),
            requires_grad=False
        )

        self.separator = Separator(num_instruments=num_instruments, checkpoint_path=checkpoint_path)

    def compute_stft(self, wav: torch.Tensor):
        """
        Computes stft feature from wav

        Args:
            wav (Tensor): B x L
        """

        L = wav.shape[-1]
        framed_wav = wav.unfold(-1, size=self.win_length, step=self.hop_length)
        framed_wav *= self.win
        stft = torch.fft.rfft(framed_wav, self.win_length)
        stft = stft.transpose(1, 2)

        # only keep freqs smaller than self.F
        stft = stft[:, :self.F, :]
        mag = stft.abs()

        return torch.view_as_real(stft), mag

    def inverse_stft(self, stft):
        """Inverses stft to wave form"""

        pad = self.win_length // 2 + 1 - stft.size(1)
        stft = F.pad(stft, (0, 0, 0, 0, 0, pad))
        stft = torch.view_as_complex(stft)
        stft = stft.transpose(1, 2)
        wav: torch.Tensor = torch.fft.irfft(stft, self.win_length)
        wav *= self.win
        wav = overlap_and_add(wav, self.hop_length)
        return wav.detach()

    def forward(self, wav):
        return self.separate(wav)

    def separate(self, wav):
        """
        Separates stereo wav into different tracks corresponding to different instruments

        Args:
            wav (tensor): 2 x L
        """

        # stft (complex tensor) - 2 X F x L
        # stft_mag - 2 X F x L
        # Compute the STFT from the mixed wav
        stft, stft_mag = self.compute_stft(wav)

        # Perform the actual stem separation
        masks = self.separator(stft_mag)

        # Recover the wavs via an inverse STFT
        wavs = []
        for mask in masks:
            stft_masked = stft * mask
            wav_masked = self.inverse_stft(stft_masked)

            wavs.append(wav_masked)

        return wavs
