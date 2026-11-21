# skyhum

**Does your audio say where you were?**

![status: alpha](https://img.shields.io/badge/status-alpha-orange)
![locator: r5b, test partial](https://img.shields.io/badge/locator-r5b%20%C2%B7%20test%20partial-yellow)
![find confidence: not validated](https://img.shields.io/badge/find%20confidence-not%20validated-red)
![licence: MIT](https://img.shields.io/badge/licence-MIT-blue)

Aircraft hum slides in pitch as a plane passes. Matched against public flight history, that slide can narrow down
where a recording was made. `skyhum` tests that on audio you give it, so you can see what your recordings and
calls leak and check whether a fix works.

- **Verify**: is this audio consistent with a place and time someone claims?
- **Audit**: you know where you were; how much did the audio give away, and does scrubbing help?
- **Scrub**: remove the clue before you share.
- **Find**: rank places inside an area you supply (research and validation).
- **Locate** (`skyhum locate --country US|AU`, coming in this release series): the whole-country search used for the write-up, so every number can be checked. Not the default.

It runs locally. Audio never leaves your machine.

**Paper / article:** https://x.com/bojie_li/status/2124169630596137359

> **Read this first.** A blind one-hour search names the right region for about 5% of public recordings. The
> risk comes from repeated sessions from one spot (rural Australian sensors reached 93% with eight sessions) and
> from a listener who already has a guess. An `UNCORROBORATED` result is not proof you are safe. See
> [Limits](#limits).

## Status of this build

This is an alpha. The command line, the place grammar, the verdicts, the reports and the safety gate are final in
shape. `verify`, `audit` and `find` use the round-5b locator from the research code (`--matcher r5b`, the
default when available). It runs on a GPU when there is one and falls back to the CPU, where it is much slower:
minutes for a short clip, far longer for an hour of audio. A simplified numpy matcher is kept as
`--matcher reference` for quick checks; it has not been benchmarked.

The research models ship in `src/skyhum/models/` (detector v3, the round-5b fitted tables, the loud-event
classifier) with checksums and a [model card](src/skyhum/models/MODEL_CARD.md) that lists every training source and
its licence. Without PyTorch, `--matcher auto` (the default) uses the reference matcher and says so.

**The round-5b test is partial.** It covered 43 selected positive hours in three clock modes, without the clock
decoys, so the confidence behind `verify`'s operating points and `find`'s point-estimate threshold is **not
validated**. See [Limits](#limits) and [docs/locator.md](docs/locator.md).

| Command | State in this build |
|---|---|
| `selftest`, `doctor`, `regions list` | working (`selftest --quick` skips the slow r5b checks) |
| `scrub` | working: the pause gate and the tone scrubber from the research code, unchanged |
| `detect` | working, with a classic ridge detector; the learned detector is pending |
| `verify` | working with r5b (default) or the reference matcher; clock windows up to ±30 min; you supply the ADS-B file |
| `audit` | working, except the exposure tier, which is not computed yet |
| `find` | gate complete; r5b ranking; research use, confidence not validated |
| `bench list` | working; `bench fetch` and `bench run` are not in this build |
| `adsb fetch`, `--adsb auto` | not in this build: download ADS-B days yourself ([docs/data.md](docs/data.md)) |

Anything not in this build exits with code 2 and says so.

## Install

<!-- skyhum-test: skip (needs network and a clone) -->
```bash
git clone https://github.com/bojieli/skyhum && cd skyhum
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only PyTorch for the r5b locator
pip install .                 # Python 3.10+
pip install ".[plot]"         # optional: spectrogram PNGs
```
ffmpeg is optional: it is needed for mp3, m4a and webm input. There is no PyPI package yet.

Check the install:
```bash
skyhum doctor                 # Python packages, ffmpeg, GPU, ADS-B cache
skyhum selftest --quick       # synthetic plane + synthetic track; should print PASS in seconds
```

## Try it on synthetic data

`selftest --write-fixtures` writes a small synthetic world: three propeller passes over a made-up site inside the
"Los Angeles basin" box, the matching synthetic ADS-B tracks, the same audio with a speech-like buzz on top, and a
noise-only clip. Nothing in it is real. On a laptop CPU the r5b steps take a few minutes; `--fast` and
`--decoys 5` keep them short and are for demos only. Step 4 uses the quick reference matcher.

```bash
skyhum selftest --write-fixtures demo

# 1. What is in the audio?
skyhum detect demo/clip.wav --start 2026-09-08T14:03:00Z

# 2. Someone says this was recorded here. Is it consistent?
skyhum verify demo/clip.wav --start 2026-09-08T14:03:00Z --adsb demo/tracks.npz \
       --claim 34.05,-118.30,r=2km --fast --decoys 5

# 3. The same audio against a place 18 km east
skyhum verify demo/clip.wav --start 2026-09-08T14:03:00Z --adsb demo/tracks.npz \
       --claim 34.05,-118.10,r=2km --fast --decoys 5          # exit 10 or 11

# 4. I know where I was. How much did it leak, and does the pause gate help?
skyhum audit demo/call.wav --start 2026-09-08T14:03:00Z --adsb demo/tracks.npz \
       --where 34.05,-118.30,r=1km --scrub demo/call_clean.wav --matcher reference

# 5. Remove the clue
skyhum scrub demo/call.wav demo/call_tones.wav --method tones

# 6. Rank places inside an area (reference matcher for speed)
skyhum find demo/clip.wav --start 2026-09-08T14:03:00Z --adsb demo/tracks.npz \
       --area 34.05,-118.30,r=4km --matcher reference          # exit 0 or 10
```
Output of step 2 (about six minutes on a busy server CPU; without PyTorch the reference matcher answers in
seconds, with the same verdict):
```
verify           clip.wav
claim            34.05,-118.30,r=2km
clock window     14:02:30Z-14:03:30Z
matcher          r5b round-5b locator (test partial; confidence not validated) [--fast]
passes used      3
decoy rank       1 of 6  (margin 24.237)
verdict          SUPPORTED  (default operating point)
why              the claim beats every decoy (5 decoy clocks)
not tested       indoor attenuation, speech overlap, platform processing
```

## Reproducing the article

Every number in the article (https://x.com/bojie_li/status/2124169630596137359) is meant to be checkable with this repository: the models it used ship in
`src/skyhum/models/`, and a reproduction guide with one command per published row is being added (`REPRODUCING.md`,
`skyhum bench run --reproduce`). Rows built on event and speech livestreams are not reproducible by third parties by
design: those recordings could identify private individuals and are not published in any form.

## With your own recordings

You need the recording, its start time in UTC, and an ADS-B day file for the area (`tracks.npz`, see
[docs/data.md](docs/data.md)).

<!-- skyhum-test: skip (needs real audio and ADS-B data) -->
```bash
skyhum detect stream.wav --start 2026-09-08T14:03:00Z --png stream.png
skyhum audit stream.wav --start 2026-09-08T14:03:00Z --adsb day.npz --where 37.80,-122.41,r=1km
skyhum audit stream.wav --start 2026-09-08T14:03:00Z --adsb day.npz --where 37.80,-122.41,r=1km \
       --scrub stream_clean.wav
skyhum verify clip.wav --start 2026-09-08T14:03:00Z --adsb day.npz --claim "Los Angeles basin" \
       --report report.html
```
Without `--fast`, r5b runs the frozen round-5b configuration and 20 decoy clocks, one locator run each. On a CPU
that is slow; use a GPU (`--device cuda`) for hour-long audio or metro-sized areas, or a small circle around the
place you want to test.

## Parameters you will use

| Parameter | Meaning | Notes |
|---|---|---|
| `--start` | When the recording began, UTC (`2026-09-08T14:03:00Z`) or epoch | Required. `--tz` for local times |
| `--start-tol` | How sure you are of the start | `5s`, `30s` (default), up to `30m` in this build. Wider = weaker evidence |
| `--claim` / `--where` / `--area` | A place | Named region, `lat,lon,r=5km`, `bbox:w,s,e,n`, or `@file.geojson` |
| `--operating-point` | How strict `verify` is | `default` or `strict` (a fixed margin; not calibrated) |
| `--matcher` | Which matcher | `r5b` (default) or `reference` |
| `--versus` | A rival place to compare against | Scored as one more decoy |
| `--duration` | Use part of the audio | |
| `--adsb` | ADS-B day file | `tracks.npz`; `auto` is not in this build |
| `--report` | Write `json` or self-contained `html` | |
| `--json` | One JSON document on stdout | |

Exit codes: `0` supported (or success) · `10` uncorroborated · `11` inconclusive · `2` usage, or not in this build ·
`3` missing data · `1` selftest failure. Full reference: [docs/cli.md](docs/cli.md).

## The modes

### verify
Scores the claimed place at the stated clock against decoys: the same place at wrong clocks (tens of minutes off)
and on other days, plus an optional rival place. The claim is `SUPPORTED` only if it beats every decoy.

| Verdict | Meaning |
|---|---|
| `SUPPORTED` | Audio and flight records fit the claim better than every decoy |
| `UNCORROBORATED` | Decoys fit as well or better. **Not** evidence the claim is false |
| `INCONCLUSIVE` | Too little to test: no aircraft in range, audio too short, ADS-B too short for decoys |

How often the earlier research scorer's default rule fired on single-hour US webcam recordings (110): it accepted
18 true claims (16%) and 2 wrong neighbours (1.8%). A neighbouring region can pass because listeners on both sides
of a boundary hear much the same plane. This has not been re-measured with r5b.

### audit
You provide the truth. The report shows the passes the detector finds, the ADS-B passes in range, whether `verify`
would accept the true place at both operating points, whether a place 15 km away also passes, and advice from fixed
rules. With `--scrub` it repeats everything on the scrubbed audio. The exposure tier (how close a search could
get) is not computed in this build.

### scrub
<!-- skyhum-test: skip (in.wav is a placeholder) -->
```bash
skyhum scrub in.wav out.wav --method gate    # replace the low band with comfort noise while nobody speaks
skyhum scrub in.wav out.wav --method tones   # track and remove long tonal lines, under speech too
```
Offline simulation on 53 quiet outdoor hours (local area and clock supplied): untreated 32 within 2 km; pause
gate 4; tone scrubbing 17; re-encoding 32. No speech cost on two speech-quality tests. Real platform pipelines may
behave differently. Re-encoding alone does nothing.

### find
`find` ranks places inside an area you supply and prints a point estimate only when the result passes the strict
threshold; otherwise it shows 0.1-degree tiles or nothing. **That threshold is not validated for r5b**: it is a fixed
margin over the decoy clocks, and how often an answer above it is wrong has not been measured. `find` says so in its
output. `--single-event` (clock known within 5 s) allows answers from one pass and prints the mirror point across
the aircraft's track as well.

## Limits
- One quiet hour rarely names a city: 15 of 327 public recordings (5%), against 50 of 327 (15%) for always
  guessing "New York metro".
- Calls: with no added voice, 17 of 19 known flybys were still recognised after Telegram Desktop and 13 of 19
  after Google Meet. With someone reading aloud 60% of the time, 0 of 19 after either app.
- Modelled exposure (for example 54% within 2 km at Lake Balboa by KVNY, 58% at The Battery) **assumes every
  audible pass is detected**. The blind detector caught 34 of 86 audible passes (40%) at three cameras.
- Tested on outdoor public webcams and research sensors, fed digitally. There is no measured success rate for an
  ordinary indoor or mobile caller.
- Needs an accurate start time. Wider clocks mean weaker evidence.
- Jets alone, and distance cues beyond ground reflection, did not help.
- **Round-5b locator (default here), partial test** on 43 selected positive hours (split_v2 test, 7 site
  clusters, with detector hints this package does not ship): within 2 km, blind 22 (round 5a: 15), clock known to
  30 min 23 (19), clock known to 30 s 31 (23). The confidence intervals overlap. The no-box mode had only 9
  positives and is not a result. The clock decoys of this test and the decoys for single-event hard and negative
  hours did not run, so the false-fix rate and the calibration of the `find` threshold are **not measured**.
  Multi-session scoring was not run.
- The numbers above the r5b line come from earlier rounds of the research scorer. `--matcher reference` has not
  been benchmarked.

More in [docs/limits.md](docs/limits.md).

## Responsible use
skyhum is for research, validation and defence: checking your own media, auditing exposure, and checking the
published numbers. **It is not for locating private individuals.** That is a statement, not a technical gate.
Verify and audit are the primary paths; audio is never uploaded; and the repository carries no recordings.
See [docs/ETHICS.md](docs/ETHICS.md).

If you run a meeting platform or a streaming product, [docs/platforms.md](docs/platforms.md) lists what we tested
and what reduced the clue.

## Documentation
[Quickstart](docs/quickstart.md) · [Concepts](docs/concepts.md) · [CLI reference](docs/cli.md) ·
[Data and licences](docs/data.md) · [Limits](docs/limits.md) · [Ethics](docs/ETHICS.md) ·
[Platforms](docs/platforms.md) · [Benchmark](docs/bench.md)

## Contributing, licence, credits
Issues and pull requests are welcome; run `pytest` and `skyhum selftest` first, and read
[CONTRIBUTING.md](CONTRIBUTING.md). Security reports: [SECURITY.md](SECURITY.md).

Licence: MIT (see `LICENSE`). Third-party code and data notices: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

skyhum builds on the physics in
[AlanBacker/audio-flyover-geolocation](https://github.com/AlanBacker/audio-flyover-geolocation) (MIT), which
implements colsto's method ([video](https://www.youtube.com/watch?v=UytYrhYLOKI)). ADS-B history from
[adsb.lol](https://adsb.lol) is ODbL.
