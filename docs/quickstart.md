# Quickstart

## 1. Install

```
git clone https://github.com/bojieli/skyhum && cd skyhum
python -m venv .venv && . .venv/bin/activate
pip install .                 # numpy, scipy, soundfile, webrtcvad-wheels
pip install ".[plot]"         # optional: spectrogram PNGs (matplotlib)
```
Python 3.10 or newer. Install ffmpeg if you want mp3, m4a or webm input; wav, flac and ogg work without it.

```
skyhum doctor       # what this install can do
skyhum selftest     # synthetic plane + synthetic track; prints PASS in a few seconds
```

## 2. Try the synthetic demo

```
skyhum selftest --write-fixtures demo
skyhum verify demo/clip.wav --start 2026-09-08T14:03:00Z --adsb demo/tracks.npz --claim 34.05,-118.30,r=2km
```
The README walks through the rest of the demo: detect, a displaced claim, audit with scrubbing, and the `find`
gate.

## 3. Your own recording

You need three things:

1. **The audio.** Any length from one minute up; an hour is typical. The first channel is used unless you pass
   `--channel`.
2. **The start time in UTC**, as exactly as you can. `--start 2026-09-08T14:03:00Z`, or a local time with `--tz`.
   Say how sure you are with `--start-tol` (default 30 s). A wider window gives chance matches more room.
3. **ADS-B history** for the area and that day, as a `tracks.npz` file. See [data.md](data.md) for how to make one.
   It must cover the recording plus about an hour either side, because the decoys need it.

Then, for audio you recorded yourself:
```
skyhum audit my.wav --start 2026-09-08T14:03:00Z --adsb day.npz --where 37.80,-122.41,r=1km --report audit.html
```
and to test a fix:
```
skyhum audit my.wav --start 2026-09-08T14:03:00Z --adsb day.npz --where 37.80,-122.41,r=1km --scrub my_clean.wav
```

## 4. Read the result

- `SUPPORTED` (exit 0): the audio fits the place better than every decoy clock.
- `UNCORROBORATED` (exit 10): it does not. This is **not** proof the place is wrong, or that you are safe.
- `INCONCLUSIVE` (exit 11): too little to test, usually no aircraft in range or ADS-B too short for decoys.

[concepts.md](concepts.md) explains what the tool tests and what it cannot say. [limits.md](limits.md) has the
measured numbers.
