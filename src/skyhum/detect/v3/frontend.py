"""Detector front end: 240 s of 1 kHz audio -> whitened log-power on 128 log-frequency bins x 300 time steps.

Log-power STFT (1 s Hann window zero-padded to 2048, 0.2 s hop), linearly interpolated onto 128 log-spaced bins
from 30 to 400 Hz, whitened twice (against the median of +-8 neighbouring bins in the same frame, then against each
bin's median over time), pooled to 300 steps of 0.8 s and scaled to roughly [-0.5, 2]. Unchanged from the research
front end except that the -60 dB dither draws from a seeded generator, so a run is reproducible.
"""
import numpy as np

FS = 1000                       # the detector only looks at 30-400 Hz
SEG_S = 240.0
N = int(FS * SEG_S)
HOP = 200                       # 0.2 s
NFFT = 2048                     # zero-padded 1 s window -> 0.49 Hz bins
WIN = 1000
F_LO, F_HI, NF = 30.0, 400.0, 128
NT = 300                        # time steps after pooling (0.8 s each)
LOGF = np.geomspace(F_LO, F_HI, NF)
HF, HT = NF // 4, NT // 4       # heatmap size


class Frontend:
    def __init__(self, dev="cpu", seed=0):
        import torch
        self.torch, self.dev = torch, torch.device(dev)
        self.window = torch.hann_window(WIN, device=self.dev)
        fr = np.fft.rfftfreq(NFFT, 1 / FS)
        M = np.zeros((NF, fr.size), np.float32)          # linear interpolation onto log-frequency bins
        idx = np.interp(LOGF, fr, np.arange(fr.size))
        i0 = np.floor(idx).astype(int)
        w = idx - i0
        M[np.arange(NF), i0] = 1 - w
        M[np.arange(NF), i0 + 1] = w
        self.M = torch.as_tensor(M, device=self.dev)
        self.gen = torch.Generator(device=self.dev)
        self.reseed(seed)

    def reseed(self, seed):
        self.gen.manual_seed(int(seed))

    def __call__(self, x):
        """x: (B, N) float tensor -> (B, 1, NF, NT) whitened log-power."""
        torch = self.torch
        # -60 dB dither: edited recordings contain digital silence, which wrecks the median whitening
        noise = torch.randn(x.shape, generator=self.gen, dtype=x.dtype, device=x.device)
        x = x + 1e-3 * x.std(dim=1, keepdim=True).clamp(min=1e-6) * noise
        S = torch.stft(x, NFFT, hop_length=HOP, win_length=WIN, window=self.window, center=True,
                       return_complex=True)
        P = S.real ** 2 + S.imag ** 2                    # (B, F, T)
        P = torch.einsum("kf,bft->bkt", self.M, P)
        D = 10 * torch.log10(P + 1e-12)
        # 1) local spectral whitening: dB above the median of +-8 neighbouring log bins in the same frame
        Dp = torch.nn.functional.pad(D.transpose(1, 2), (8, 8), mode="replicate").transpose(1, 2)
        D = D - Dp.unfold(1, 17, 1).median(dim=-1).values
        # 2) remove steady tones: subtract each bin's median over time
        D = D - D.median(dim=2, keepdim=True).values
        D = torch.nn.functional.adaptive_avg_pool1d(D, NT)
        return (D.clamp(-5, 20) / 10.0).unsqueeze(1)
