# Limits

Every number on this page comes from the research evaluations. The round-5b locator (the package default) has a
partial test, summarised below and in [locator.md](locator.md); the other numbers come from earlier rounds.
`--matcher reference` has not been benchmarked.

## Round-5b locator (partial test)

On 43 selected positive hours (7 site clusters, with detector hints the package does not ship), top-1 within 2 km:
blind 22 (round 5a: 15), clock known to 30 min 23 (19), clock known to 30 s 31 (23). The confidence intervals
overlap. The wide mode had 9 positives and is not a result. The clock decoys, the single-event decoys and the
multi-session test did not run: **the false-fix rate and the calibration of the strict threshold used by `verify`
and `find` are not measured.**

## How often a location comes out

- **Blind search, one quiet hour:** the right region for 15 of 327 public recordings (5%). Always answering
  "New York metro" scores 50 of 327 (15%). One recording, searched blind, rarely names a city.
- **Repetition is the real risk.** Rural Australian sensors reached 93% with eight sessions from the same spot.
  Several sessions from one place add up even when each alone says little.
- **A listener with a guess.** Checking a claim is easier than finding a place. On single-hour US webcam
  recordings (110), the default claim check accepted 18 true claims (16%) and 2 wrong neighbouring claims (1.8%).

## What the modelled numbers assume

Modelled exposure, for example 54% within 2 km at Lake Balboa by KVNY and 58% at The Battery, **assumes every
audible pass is detected**. In practice the blind detector caught 34 of 86 audible passes (40%) at three cameras.
Treat modelled tiers as an upper bound and measured results as what happened on that audio.

## Calls

With no added voice, 17 of 19 known flybys were still recognised after Telegram Desktop and 13 of 19 after Google
Meet. With someone reading aloud 60% of the time, 0 of 19 after either app. See [platforms.md](platforms.md).

## Scrubbing (offline simulation)

On 53 quiet outdoor hours, with the local area and clock supplied: untreated, 32 located within 2 km; with the pause
gate, 4; with tone scrubbing, 17; after re-encoding alone, 32. Neither scrubber cost measurable speech quality on
two speech-quality tests. This is a simulation on recorded audio, not a measurement through a real platform.

## Conditions not covered

- Tested on outdoor public webcams and research sensors, fed digitally. There is no measured success rate for an
  ordinary indoor or mobile caller.
- The start time must be known. A wider clock window means weaker evidence.
- Jets alone did not help, and distance cues beyond ground reflection did not help.
- Only aircraft that broadcast ADS-B and appear in the archive can be matched.

## Limits of this build

- Default matcher r5b: slow on CPU; 20 decoys for verify and 10 for find by default; run without detector hints.
  `--matcher reference`: not benchmarked; 0.5 km cells, coarser for large areas (flagged).
- Clock windows up to ±30 min.
- Strict operating point: a fixed margin, not calibrated to a false-accept rate.
- Terrain: flat ground at `--ground-alt` unless `--dem-dir` points at GeoTIFF tiles.
- Detector: the classic ridge detector; the learned detector is pending.
- `audit` does not compute exposure tiers yet. `bench fetch`, `bench run`, `adsb fetch` and `--adsb auto` are not
  implemented.
