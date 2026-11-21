# Concepts

## The Doppler clue

A propeller aircraft or helicopter makes a steady tone: blades passing at a fixed rate, plus harmonics. A listener
on the ground hears that tone higher as the aircraft approaches and lower as it leaves. The pitch slides most
steeply at the closest point of approach, and how steep the slide is depends on the aircraft's speed and how far
away it passes.

ADS-B is the position broadcast most aircraft send every second; volunteer networks archive it. Given a place and a
time, the archive says which aircraft flew near that place, and the physics says exactly what pitch curve each one
would have produced there. If the curves predicted for a place line up with ridges in the recording's spectrogram,
the recording fits that place. Curves for a place a few kilometres away have different timings and shapes.

One pass rarely means much: many places see some aircraft at some time. Two or more passes that line up at the
same clock are much harder to match by chance. That is why the matcher only scores a place when at least two
separate sounds agree.

## Far hum and near roar

Two kinds of evidence come from aircraft:

- **Far hum.** A distant propeller or helicopter, often kilometres away, leaves a faint tonal line in the low
  frequencies (roughly 30 to 600 Hz) that people in the room do not notice. Most of the location clue is here. It
  survives a lot: background noise, moderate compression, some call pipelines. It is drowned out by continuous
  speech over the same stretch.
- **Near roar.** An aircraft passing close by is loud and broadband: everyone hears it. Its timing alone carries
  some information, and it is much harder to remove: the scrubbers in this tool target tonal lines, and a loud
  pass also tends to open a speech detector, so the pause gate lets part of it through.

## What verify tests

`verify` asks one question: does the audio fit the claimed place at the stated clock better than it fits the same
place at clocks where it should not fit? Those wrong clocks are the decoys: the same window shifted by tens of
minutes, or moved to another day when the ADS-B file covers it. A decoy gets exactly the same number of chances as
the real window. A rival place given with `--versus` is one more decoy.

- `SUPPORTED`: the claim beats every decoy. At the default operating point that is the whole rule. `strict` also
  requires a margin.
- `UNCORROBORATED`: some decoy did as well. The most common reasons are ordinary: speech covering the passes, no
  close traffic, an inexact clock, audio processed by a call or upload. It is not evidence that the claim is false.
- `INCONCLUSIVE`: there was nothing to test.

## What verify cannot say

- It cannot prove where a recording was made. A neighbouring place can fit about as well, because listeners a few
  kilometres apart hear the same aircraft.
- It cannot prove a recording was not made somewhere. Most true claims are not corroborated.
- It knows only aircraft that broadcast ADS-B and are in your track file.
- It assumes the audio is a continuous recording from one place. Edited or spliced audio is out of scope.
- It does not model buildings or indoor attenuation. Terrain is flat at `--ground-alt` unless you pass `--dem-dir`.

## audit and find

`audit` runs the checks from your side: you say where you were, and it reports what was detectable, whether the
true place would be corroborated, and whether a place 15 km away would pass too.

`find` turns the question around: which cells inside a given area fit best? It is gated (see
[ETHICS.md](ETHICS.md)) and capped at 100 km across, and it prints a point estimate only above the strict
threshold.

## The matcher in this build

The default is the round-5b locator from the research code ([locator.md](locator.md)): the same physics with
calibrated per-pass evidence, an f0 prior and an envelope term. Its test was partial, and the confidence of its
decisions has not been validated. `--matcher reference` selects a simplified numpy re-implementation: the same
physics and two-voter rule without the calibrated terms, candidate pruning, null or refinement. It is fast and has
not been benchmarked.
