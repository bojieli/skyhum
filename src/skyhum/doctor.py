"""doctor: report what this install can and cannot do. Read-only; touches the GPU only if asked to probe it."""
import importlib.util
import os
import platform
import shutil
import sys

from skyhum import __version__


def _ver(mod):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "installed")
    except Exception:
        return None


def run(device="auto", workdir=None):
    checks = []

    def add(name, status, detail):
        checks.append(dict(check=name, status=status, detail=detail))

    add("skyhum", "ok", __version__)
    py = sys.version_info
    add("python", "ok" if py >= (3, 10) else "fail", platform.python_version())
    for mod, need in (("numpy", True), ("scipy", True), ("soundfile", True)):
        v = _ver(mod)
        add(mod, "ok" if v else ("fail" if need else "warn"), v or "missing (pip install skyhum)")
    add("webrtcvad", "ok" if importlib.util.find_spec("webrtcvad") else "warn",
        "installed" if importlib.util.find_spec("webrtcvad") else "missing: `scrub --method gate` will not run")
    add("matplotlib", "ok" if importlib.util.find_spec("matplotlib") else "info",
        "installed" if importlib.util.find_spec("matplotlib") else "missing: no spectrogram PNG (pip install "
                                                                    "'skyhum[plot]')")
    ff = shutil.which("ffmpeg")
    add("ffmpeg", "ok" if ff else "warn", ff or "missing: only wav/flac/ogg input, no mp3/m4a/webm")
    probe = device in ("auto", "cuda") and not os.environ.get("SKYHUM_NO_GPU_PROBE")
    if importlib.util.find_spec("torch") is None:
        add("torch", "warn", "missing: the r5b locator (default matcher) will not run; pip install torch, or use "
                             "--matcher reference")
    else:
        add("torch", "ok", "installed")
    from skyhum.match import locator as LOC
    add("r5b tables", "ok" if LOC.models_available() else "warn",
        "installed" if LOC.models_available() else "not installed (not in the public repository yet): "
                                                   "--matcher auto uses the reference matcher")
    if importlib.util.find_spec("torch") is None:
        pass
    elif not probe:
        add("gpu", "info", "not probed (--device cpu or SKYHUM_NO_GPU_PROBE set); r5b runs on CPU, slowly")
    else:
        try:
            import torch
            ok = torch.cuda.is_available()
            add("gpu", "info", (torch.cuda.get_device_name(0) + ": r5b runs on it") if ok
                else "no CUDA device: r5b runs on CPU, slowly")
        except Exception as e:
            add("gpu", "warn", f"probe failed: {e}")
    cache = os.path.expanduser("~/.cache/skyhum/adsb")
    n = len(os.listdir(cache)) if os.path.isdir(cache) else 0
    add("adsb cache", "info", f"{cache}: {n} file(s); automatic download is not in this build (pass --adsb PATH)")
    add("detector weights", "info", "none needed: this build uses the classic ridge detector; the learned "
                                    "detector is pending")
    if workdir:
        add("workdir", "ok" if os.access(os.path.dirname(os.path.abspath(workdir)) or ".", os.W_OK) else "warn",
            workdir)
    status = "fail" if any(c["status"] == "fail" for c in checks) else "ok"
    return dict(version=__version__, status=status, checks=checks)
