import torch
import torch.nn.functional as F

from pathlib import Path
from torch import nn

from spleeter_pytorch.separator import Separator

class Estimator(nn.Module):
    def __init__(self, num_instruments: int, checkpoint_path: Path):
        super().__init__()

        # stft config
        self.F = 1024
        self.T = 512
        self.win_length = 4096
        self.hop_length = 1024
        self.win = nn.Parameter(
            torch.hann_window(self.win_length),
            requires_grad=False
        )

        self.separator = Separator(num_instruments=num_instruments, checkpoint_path=checkpoint_path)

    def compute_stft(self, wav):
        """
        Computes stft feature from wav

        Args:
            wav (Tensor): B x L
        """

        stft = torch.stft(wav, n_fft=self.win_length, hop_length=self.hop_length, window=self.win,
                          center=True, return_complex=True, pad_mode='constant')
        
        # implement torch.view_as_real(stft) manually since coremltools doesn't support it
        stft = torch.stack((torch.real(stft), torch.imag(stft)), axis=-1)

        # only keep freqs smaller than self.F
        stft = stft[:, :self.F]

        # implement torch.hypot manually since coremltools doesn't support it
        mag = torch.sqrt(stft[..., 0] ** 2 + stft[..., 1] ** 2)

        return stft, mag

    def inverse_stft(self, stft):
        """Inverses stft to wave form"""

        pad = self.win_length // 2 + 1 - stft.size(1)
        stft = F.pad(stft, (0, 0, 0, 0, 0, pad))

        # implement torch.view_as_complex(stft) manually since coremltools doesn't support it
        stft = stft[..., 0] + stft[..., 1] * 1j

        wav = torch.istft(stft, self.win_length, hop_length=self.hop_length, center=True,
                    window=self.win)
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
