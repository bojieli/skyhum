"""Audio decode, channel pick, resample and trim. Everything stays in memory on this machine."""
import os
import shutil
import subprocess
from math import gcd

import numpy as np

from skyhum.errors import MissingData, UsageError

FS = 16000
NATIVE = {".wav", ".flac", ".ogg", ".aiff", ".aif"}


def ffmpeg_path():
    return os.environ.get("SKYHUM_FFMPEG") or shutil.which("ffmpeg")


def load(path, channel=0, fs=FS, duration_s=None, offset_s=0.0):
    """Return (x float32 mono at fs, info dict)."""
    if not os.path.isfile(path):
        raise UsageError(f"audio file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    x = sr = None
    if ext in NATIVE:
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=True)
            if channel >= data.shape[1]:
                raise UsageError(f"{path} has {data.shape[1]} channel(s); --channel {channel} does not exist")
            x = data[:, channel]
        except ImportError:
            x = None
    if x is None:
        ff = ffmpeg_path()
        if not ff:
            raise MissingData(f"decoding {ext or 'this file'} needs ffmpeg on PATH (or install soundfile for wav/flac)")
        cmd = [ff, "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0",
               "-af", f"pan=mono|c0=c{int(channel)}", "-ar", str(fs), "-f", "f32le", "pipe:1"]
        p = subprocess.run(cmd, capture_output=True)
        if p.returncode != 0:
            raise UsageError(f"ffmpeg could not decode {path}: {p.stderr.decode(errors='replace').strip()[:300]}")
        x, sr = np.frombuffer(p.stdout, np.float32).copy(), fs
    x = resample(np.asarray(x, np.float32), int(sr), fs)
    a = int(round(offset_s * fs))
    b = x.size if duration_s is None else min(x.size, a + int(round(duration_s * fs)))
    x = x[a:b]
    if x.size == 0:
        raise UsageError(f"{path}: no audio after --duration/offset")
    return x, dict(path=os.path.abspath(path), source_rate=int(sr), rate=fs, channel=int(channel),
                   duration_s=round(x.size / fs, 3))


def resample(x, sr_in, sr_out):
    if sr_in == sr_out:
        return x
    from scipy.signal import resample_poly
    g = gcd(int(sr_in), int(sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def save(path, x, fs=FS):
    """Write wav/flac with soundfile; anything else through ffmpeg."""
    ext = os.path.splitext(path)[1].lower()
    x = np.clip(np.asarray(x, np.float32), -1.0, 1.0)
    if ext in (".wav", ".flac"):
        import soundfile as sf
        sf.write(path, x, fs, subtype="PCM_16" if ext == ".wav" else None)
        return
    ff = ffmpeg_path()
    if not ff:
        raise MissingData(f"writing {ext} needs ffmpeg; write .wav or .flac instead")
    subprocess.run([ff, "-v", "error", "-nostdin", "-y", "-f", "f32le", "-ar", str(fs), "-ac", "1", "-i", "pipe:0",
                    path], input=x.tobytes(), check=True, capture_output=True)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def file_start_time(path, duration_s):
    """Best guess of the start from container metadata (creation_time) or mtime minus duration."""
    ff = shutil.which("ffprobe")
    if ff:
        p = subprocess.run([ff, "-v", "error", "-show_entries", "format_tags=creation_time", "-of",
                            "default=nw=1:nk=1", path], capture_output=True, text=True)
        val = p.stdout.strip()
        if p.returncode == 0 and val:
            from skyhum.io.clock import parse_start
            try:
                return parse_start(val), "container creation_time"
            except Exception:
                pass
    return os.path.getmtime(path) - duration_s, "file mtime minus duration"
