import torch
import torch.nn.functional as F

from pathlib import Path
from torch import nn

from spleeter_pytorch.separator import Separator
from spleeter_pytorch.stft import STFT
from spleeter_pytorch.util import overlap_and_add

class Estimator(nn.Module):
    def __init__(
        self,
        num_instruments: int,
        checkpoint_path: Path,
        use_torch_stft: bool=True,
    ):
        super().__init__()

        # stft config
        self.F = 1024
        self.T = 512
        self.win_length = 4096 # should be a power of two, see https://github.com/tensorflow/tensorflow/blob/6935c8f706dde1906e388b3142906c92cdcc36db/tensorflow/python/ops/signal/spectral_ops.py#L48-L49
        self.hop_length = 1024
        win_func = torch.hann_window
        self.win = nn.Parameter(
            win_func(self.win_length),
            requires_grad=False
        )

        self.separator = Separator(num_instruments=num_instruments, checkpoint_path=checkpoint_path)
        self.use_torch_stft = use_torch_stft

        if not use_torch_stft:
            self.stft = STFT(
                filter_length=self.win_length,
                hop_length=self.hop_length,
                win_length=self.win_length,
                win_func=win_func,
            )

    def compute_stft(self, wav: torch.Tensor):
        """
        Computes stft feature from wav

        Args:
            wav (Tensor): B x L
        """

        if self.use_torch_stft:
            stft = torch.stft(
                wav,
                n_fft=self.win_length,
                hop_length=self.hop_length,
                window=self.win,
                center=True,
                return_complex=True,
                pad_mode='constant'
            )
        else:
            stft = self.stft.transform(wav)

        # only keep freqs smaller than self.F
        stft = stft[:, :self.F, :]
        mag = stft.abs()

        return torch.view_as_real(stft), mag

    def inverse_stft(self, stft):
        """Inverses stft to wave form"""

        pad = self.win_length // 2 + 1 - stft.size(1)
        stft = F.pad(stft, (0, 0, 0, 0, 0, pad))
        stft = torch.view_as_complex(stft)
        if self.use_torch_stft:
            wav = torch.istft(
                stft,
                self.win_length,
                hop_length=self.hop_length,
                center=True,
                window=self.win
            )
        else:
            wav = self.stft.inverse(stft)
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
