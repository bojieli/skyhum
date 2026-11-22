# Data card: webcam features

This card describes the derived features of public webcam audio that back the published webcam rows. The audio is
not released, now or later. What is released is the numbers the research pipeline consumed from it, computed by
`skyhum.bench.features`. Anyone who has the audio can recompute them with
`python -m skyhum.bench.features extract`.

The features are **not in this repository**. They are staged as release assets: tar shards under 2 GB each, a
`SHA256SUMS` file and this card. The repository holds the hour list (`src/skyhum/data/webcam_features.json`: video id,
URL, camera, place, start UTC, duration, rows) and the code that writes and reads the files.

## Status of each set

| set | contents | band | status |
|---|---|---|---|
| **3, restricted** (`tonal_400` + `loud_env`) | locator spectrogram rows up to 400 Hz; two 2 s level curves | 29.8-400 Hz; level curves over 25 Hz-6.3 kHz and 50-180 Hz | released (owner, 2026-11-21) |
| **1, tonal full** (`tonal_620`) | the locator spectrogram as it consumes it | 29.8-620 Hz | released (owner, 2026-11-22) |
| **2, loud full** (`loud_full`) | third-octave levels and frame features as the research detector stores them | 25 Hz-6.3 kHz bands (band edges 22 Hz-7.1 kHz) | released (owner, 2026-11-22) |
| 2, per-event measurements (`loud_events`) | the research detector's per-event measurements | – | **not released** (owner, 2026-11-22) |

Sets 1, 2 (`loud_full`) and 3 are released as assets of the `v0.2.0-alpha` GitHub release, with per-file
checksums, a combined `SHA256SUMS` and this card. `loud_events` is not released.

## Hours

570 hour files from 40 public cameras (554 US, 16 UK), 570 h of audio. 522 are original
recordings. 48 are the call-test copies of 16 hours: unprocessed, Google Meet and Telegram, 16 each. Hours per row
(one hour can serve several rows):

| row | hours |
|---|---|
| B (country search; plus one Midway hour staged without a 60-min window, also used by B7 / J5 / K) | 110 |
| B7 (real-world run) | 136 |
| J city / J airport / J5 Midway dev | 383 / 16 / 16 |
| K (relabelled hearing curve) | 149 |
| round-5b test: webcam positives / controls | 5 / 6 |
| E2/F2 (call test v3) | 48 |

Each entry in `src/skyhum/data/webcam_features.json` lists the video id, URL, camera slug, place and nearest airfield,
country, the camera's public location, start UTC, duration, condition, rows, `speech_share` and `loud_events`.

The rows are the published result tables: B (country-wide metro search on webcams, 110 one-hour units at 14 sites),
B7 (round-5a real-world run, 136 webcam recordings), J (low jets: 383 city and 16 airport test units, plus the 16
Midway dev hours of J5), K (the relabelled urban hearing curve: 149 webcam hours, Red Hook included), the round-5b
test (its 5 webcam positives and 6 webcam controls), and E2/F2 (call test v3: 4 cameras x 4 hours, each hour as the
unprocessed copy, received over Google Meet and received over Telegram Desktop).

## What was left out, and why

* **Events and speech livestreams.** None are included: no rallies, parades, protests, talks, amateur-radio
  activations or other speech streams. These sets were used next to the webcams in the same studies: CONUS benchmark
  events 153 and speech 88 sessions (the same recordings as J's 130 event and 79 speech units), and round-5a
  real-world events 200 and speech 92 recordings. The other non-webcam sources (xeno-canto 205, one other field
  recording) are also out; they have their own licences.
* **Private streams.** The registry flags two streams as private (a family stream and a privately run home camera).
  No listed row used them, and nothing of either is released or listed; they are not named here. A keyword search of all 241 registry entries for private / home / family / residence found no
  other private stream among the 40 cameras used. Hits on "house" were Belen's Harvey House depot, lighthouses and a
  lakeside inn.
* **A2O hours.** Not included: the audio is re-downloadable under CC BY 4.0.
* **Hours outside the reported numbers.** 16 J sessions (cheyenne_vr and belen_vr, 8 each) had no ADS-B and are not
  in J's units. 8 Midway 2026 dev sessions are not among J5's hours. None of these 24 is released.
* **Speech inside webcam hours (flagged, not removed).** An AudioSet tagger has scored 218 of the 570 hours. 32 have
  speech in >= 50 % of 10 s windows: LGA 11, RNO 11, Lake Hood 9, Tucson 1. This is airport radio (ATC) audible on the
  stream, and narration on the archived 2023 Lake Hood hours. They stay in, because the B7, K and J airport rows use
  them. `speech_share` marks them in the hour list. The 352 untagged hours were not screened for speech.

## Construction

All arrays are real numbers stored as float16 in compressed `.npz` files (one per hour per set). Axes are float32.
A JSON header (`header`, as uint8 bytes) records the set, band, window, hop, sample rate, camera, condition and start
time. No file holds a complex value, a phase, or a waveform sample. `skyhum.bench.features.load` checks this
and the band on every read.

**Tonal (`tonal_620`, `tonal_400`).** Audio decoded and resampled to 4 kHz by ffmpeg. Short-time power spectrum with
a 1 s periodic Hann window, 0.2 s hop and 4x zero padding (bin spacing 0.244 Hz). Each bin is put in dB and has a
slowly varying background removed: the median over 120 s blocks, interpolated, then the per-frame median across
frequency. Frames dominated by broadband transients (speech, door slams, wind gusts) are set to 0. The result is clipped
to 0-15 dB, which is what the locator does on load, so the clip removes nothing the locator reads. This is
`afg.audio.compute_spectrogram(x, 4000, win_s=1.0, hop_s=0.2, fmax=620)`. Rows run from 29.8 Hz, the one bin
below 30 Hz that the locator interpolates from, to 620.1 Hz (`tonal_620`) or to 400 Hz (`tonal_400`). The two sets
are the same array: `tonal_400` is the first 1,517 rows of `tonal_620`. The 400 Hz cut is applied after whitening
and gating, so the rows that remain are identical.

**Loud, full (`loud_full`).** Audio at 16 kHz, or 8 kHz when the source is below 16 kHz. STFT power with a 128 ms
window and 32 ms hop, averaged to 0.256 s frames. Arrays:

* `L`: 25 third-octave band levels in dB, centres 25 Hz-6.3 kHz. A band whose upper edge is above 0.45 x the
  sample rate is NaN.
* `Lf`: floor-tracked band levels: per band, the 10th percentile of the 32 ms powers over +-0.75 s.
* `gf_*` / `gm_*`: floor-tracked and mean levels of four band groups (50-180 Hz, 180-900 Hz, 180-3600 Hz, 900-3600 Hz).
* `flat`: spectral flatness over 400-3400 Hz.
* `crest_lf`, `crest_hf`: tonal crest (max minus median, dB) over 40-400 Hz and 400-3400 Hz.
* `modp`, `modf`, `modh`: amplitude-modulation peak prominence, frequency (9-40 Hz) and second-harmonic level of the
  300-3400 Hz envelope, one value per frame.

Where the research store held these arrays (218 hours), they are released as stored. The others
(352 hours) were recomputed by the numpy port in this package. On a test hour the port matches the
research code run on CPU to float16 rounding, at most 0.016 dB. The stored files were computed on a GPU and differ
from both in the modulation features: `modh` by more than 0.05 dB in 18 % of frames, from float32 rounding and
argmax ties. `loud_events` holds the research detector's per-event measurements (JSON, 522 hours). These
are the loud classifier's actual inputs. They include a high-frequency Doppler fit that the frame arrays cannot
reproduce. They come from the research code and cannot be recomputed by this package.

**Loud, restricted (`loud_env`).** From `L`: band powers summed over all present bands (`env_all`) and over
50-180 Hz (`env_low`), each a 2.048 s moving power mean (8 frames), in dB, sampled every 0.512 s. The value at index i
is centred at `t0_s` + 0.512 i seconds.

**Reading.** `skyhum.bench.features.load(path)` returns the arrays as float32. `locator_spec(item, path)` writes the
locator's `--spec` input. For `tonal_400` it fills 400-620 Hz with 0, the background level after the clip.
`skyhum.score.r5b_rule.Session.from_features(path, tmpdir)` builds a verify / find session from a tonal file.
`loud_frames_from(item)` returns the frame dict the loud detector reads.

## Sizes

| set | files | shards | size |
|---|---|---|---|
| 3: `tonal_400` | 570 | 9 tar | 16.9 GB (about 30 MB per hour) |
| 3: `loud_env` | 570 | 1 tar | 13.6 MB |
| 1: `tonal_620` | 570 | 15 tar | 26.8 GB (about 47 MB per hour) |
| 2: `loud_full` + `loud_events` | 570 + 522 | 1 + 1 tar | 695 MB + 97 MB |

Totals: set 3 16.9 GB, sets 1 + 2 27.7 GB. Every shard is under 1.9 GB. Each set has a
`*-FILES.sha256` that lists the checksum of every member, and `SHA256SUMS` covers the shards.

## Reproduction fidelity of the restricted set (the band conflict)

The locator reads 30-600 Hz. The loud classifier reads third-octave bands up to 6.3 kHz and per-event measurements.
The owner's restricted set stops at 400 Hz and keeps only two 2 s level curves. Results:

**Tonal (locator).** The round-5b locator ran through the package path (`skyhum.match.locator.run`, frozen
round-5b arguments, no detector hints, clock +-30 s, 0.25 deg box with a fixed jitter, CPU) on the same hour twice:
from `tonal_620` and from `tonal_400` padded with background above 400 Hz.

| hour (round-5b webcam positive) | top-1 cell | top-1 score | top-1 error | ranks 2-3 |
|---|---|---|---|---|
| Lake Hood 2026-11-15 18:44Z | the same cell both ways (61.2056 / 61.2052 N, 150.0034 W; 50 m apart) | 19.06 (620) vs 18.24 (400), -4 % | 3.38 vs 3.34 km | different cells (5.6 / 5.4 vs 6.8 / 5.3) |

Two Red Hook positives and one control were still running on the shared CPU when this was written, over 9 h each, and
are not reported. These runs do not reproduce the published round-5b numbers: those used detector hints and a GPU.
They are a paired check of the band cut only. On this hour the top answer survives the cut, and the score and the
lower-ranked hypotheses move. One hour is not a measurement of the rows. The country-wide B and J runs were not re-run
from features.

**Loud channel.** Webcam test events from the loud study at hours in this release: 3997 events,
114 of them close airborne passes, 107 hours. The shipped loud classifier was scored on the full
96 event features, and again with only what a 2 s level curve could supply (18 envelope-shape and level features;
the band spectrum, modulation, crest, flatness and Doppler features set to missing). This bound is optimistic: the
kept features keep their full-resolution values.

| | full features | restricted (bound) |
|---|---|---|
| AUC, P(aircraft), positives vs negatives | 0.3455 | 0.4669 |
| detections at the 0.2 / 1 / 3 false-alarms-per-hour thresholds | 1/114, 6/114, 14/114 | 0/114, 0/114, 0/114 |
| false alarms at the same thresholds | 19/3883, 202/3883, 572/3883 | 0, 0, 0 |

P(aircraft) changed by a median of 0.0238 and a 90th percentile of 0.2831. The classifier
is already weak on webcam audio: its AUC is below 0.5 here, consistent with row K7. Without the band features it
fires on nothing. Event timing partly survives: 103/114 positive events
have a peak of `env_all`, using the detector's own peak rule, within 5 s (median offset 0.06 s,
MAD 0.56 s).

**Conclusion.** Set 3 reproduces the tonal locator approximately on the hour tested. It does not reproduce the loud
classifier, so the loud-channel rows (the loud parts of J, the K loud rows, the combined channels) need set 2 to be
reproduced exactly.

## Speech and intelligibility

**Test (sets 2 and 3).** 10 one-minute stretches with speech (airport radio and narration) from 5 hours (Lake Hood
2023 x2, RNO x2, LGA). Each was reconstructed from the released features only with a noise vocoder: band-limited
noise in each band, scaled by the band level interpolated in time. It was done three ways:

* blind: naive frame times and a fixed output level;
* aligned: the original audio used to align time (+-0.6 s) and to match the level;
* chance control: features from 10 min later in the same hour, aligned and level matched the same way.

ASR was faster-whisper `base` (int8, CPU, temperature 0). The reference is the ASR of the original stretch
(577 words). Audio, reconstructions and transcripts were held in memory only. Only the numbers
were kept.

| features | WER blind | WER aligned | WER control | reference words matched (blind / aligned / control) |
|---|---|---|---|---|
| set 2, `loud_full` (25 bands, 0.256 s) | 2.343 | 2.149 | 1.43 | 49 / 29 / 17 of 577 |
| set 3, `loud_env` (2 curves, 2 s smoothing) | 1.0 | 1.0 | 1.0 | 0 / 0 / 0 of 578 |

On set 2, no stretch scored a WER below 0.98. WERs above 1 come from ASR hallucination on noise: the blind
reconstructions produced 2.1x as many words as the references. The matched words are at the chance level per word
produced: 4.0 % (blind), 2.6 % (aligned), 3.0 % (control). On set 3, ASR returned no words at all. By the test's
criterion (WER clearly below 90 %) nothing was intelligible, so the coarser-feature fallback was not needed.

**Why speech is not reconstructable at this time resolution.** Speech intelligibility lives in envelope modulations
of about 2-16 Hz, the syllable rate of about 4-5 Hz and the faster phonetic transitions. Set 2 has one level per
band every 0.256 s (3.9 Hz), and each level is the mean power over that frame. Envelope fluctuations above about
2 Hz are therefore averaged away and cannot be represented, since the Nyquist limit is 1.95 Hz. Envelope low-pass
studies (Drullman et al., 1994) found that cutting envelope modulations below about 2-4 Hz destroys most sentence
intelligibility. The remaining slow, 25-band spectral outline is a coarse vocoder-like representation: it shows
when sound is present and its rough spectral tilt, not which syllables were said. The modulation features `modp` /
`modf` / `modh` summarise the 9-40 Hz modulation of one broad band (300-3400 Hz) as one peak per frame. They carry
no per-band timing. Set 3 is coarser still: two curves smoothed over 2 s.

**Limits.** What the test covered: automatic speech recognition (faster-whisper `base`) on noise-vocoder
reconstructions of 10 stretches. What it did not cover: no listening test with people, and no learned
attacker (for example a speech model trained to invert band levels). This is an ASR test, not a listening test. No human listeners were used. An adversary with a better
reconstruction (e.g. a learned speech prior conditioned on the 25 band levels) was not tested. The stretches were
airport radio and narration, not close-talking speech.

**Set 1, speech above 400 Hz.** Test: the AudioSet tagger's speech windows vs other windows of the same 8 most
speech-heavy hours (786 speech vs 161 other 10 s windows). Result: the
400-620 Hz rows show *when* speech is present (mean level AUC 0.916, against 0.507
for the 30-400 Hz rows of set 3). They cannot carry syllable-rate structure: the 0.2 s hop puts the Nyquist limit at
2.5 Hz, below the 2-8 Hz syllable band, and in the representable 0.5-2.5 Hz range the modulation share separates
speech only weakly (AUC 0.374 vs 0.44 for 30-400 Hz). In the audio itself, the 400-620 Hz 2-8 Hz modulation share gives AUC 0.32. The value is below 0.5: in
these airport-radio hours, the speech windows have less syllable-rate modulation in this band than the other
windows do.

## Licences and terms

* **Webcam livestreams (YouTube).** The streams are under YouTube's standard licence. No audio, video or frames are
  redistributed. The release holds derived numeric features plus the video id and the time, which are enough to
  find the stream. Whether the platform terms allow publishing derived features is the owner's decision (recorded
  2026-11-21). It is not a licence grant from the stream owners.
* **adsb.lol (ODbL 1.0).** The ground truth for every result comes from adsb.lol and is not in these files. Fetch
  it from adsb.lol. Any database derived from it stays under ODbL with attribution.
* **Code.** `skyhum.bench.features` is MIT, like the rest of the package.

## Intended use

Reproducing the listed rows (B, B7, J, K, the round-5b webcam cases, E2/F2) from the features plus adsb.lol data,
and checking the locator and the loud channel on the same inputs. The features are not for identifying people,
voices or private places. The hour list contains only public airport and city cameras.
