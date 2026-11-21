# Benchmark

The benchmark is a set of public recordings with a known place and start time, each paired with the ADS-B day for
its area. The repository holds **links and metadata only**: no audio, no derived audio features.

## Sources

`skyhum bench list` prints this table.

| Source | What | Licence of the recordings |
|---|---|---|
| xeno-canto | long field recordings | per recording: mostly CC BY-NC-SA, some CC BY-NC-ND, a few CC BY or CC0 |
| Freesound | ambience recordings | per sound: CC0, CC BY or CC BY-NC |
| Australian Acoustic Observatory | rural sensors, repeated sessions | CC BY 4.0 |
| Webcams | public airport and city livestreams | platform standard licence: video id and time only |
| Public events | talks and events with known venue and time | platform standard licence: video id and time only |
| adsb.lol | ADS-B ground truth | ODbL 1.0 |

Rules for the catalogues (applied before the first public release):

- Each entry carries the original URL, the licence as stated by the host, the place and the start time.
- Recordings under a NoDerivatives licence are listed by link only. No labels or detections derived from them are
  published.
- Video sources are listed by id and time. Nothing is re-hosted.
- Entries that identify a private home or a private person are removed.

## Manifests

`src/skyhum/bench/manifests/` holds the public A2O manifests (links and metadata only, CC BY 4.0 recordings):
`a2o_country_v1.json` (139 one-hour sensor recordings of the whole-country benchmark, 13 same-sensor groups) and
`a2o_box_v1.json` (76 A2O hours of the held-out box test). Event and speech livestream items are in no public
manifest. Webcam rows need the webcam derived-feature release.

## Reproducing published rows

`skyhum bench run --reproduce FAMILY --outputs DIR ...` rescores locator outputs (yours, or the published saved
outputs) and writes the tables under `WORKDIR/reproduce/`. [REPRODUCING.md](../REPRODUCING.md) maps every published
row to its family, inputs and the number to compare against.

| Family | Rows | Inputs |
|---|---|---|
| `country` | region top-1, tiers, decoy confidence, multi-session | `--outputs`, `--truth`, static tables |
| `prior` | population / UTC-offset prior, the sound given a prior | same; needs timezonefinder |
| `verify` | verifying a claimed place | same; needs timezonefinder |
| `box` | round-5b in a jittered box, three clocks | `--outputs`, `--cases`, `--tag`, `--ref-tag` |
| `single-event` | two-point single-event answers with decoys | `--outputs`, `--cases`, `--tag` |

## Status

`bench list` and `bench run --reproduce` (rescoring) work. `bench fetch` exits with code 2: recordings are
downloaded from each manifest url by you.
