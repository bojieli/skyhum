# Benchmark

The benchmark is a set of public recordings with a known place and start time, each paired with the ADS-B day for
its area. The repository holds **links and metadata only**, never audio. For the public webcam hours behind the
published webcam rows, derived features (magnitude spectrograms and level curves, no audio) are prepared as release
assets, not repository files. [DATA_CARD.md](../DATA_CARD.md) describes them, and
`src/skyhum/data/webcam_features.json` lists the hours.

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

## Tiers

`bench run` (not in this build) will report, for any set of recordings, the share located within 100 m, 300 m,
1, 2, 5 and 10 km, and the share placed in the right metro within 25, 50 and 100 km, always with n.
`bench run --reproduce paper` will re-derive the published tables from saved results.

## Status

`bench list` works. `bench fetch` and `bench run` exit with code 2 in this build: the catalogue licence review
(which entries can be listed, and which labels can be published) comes first.
