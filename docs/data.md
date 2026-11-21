# Data, licences, and what leaves your machine

## What leaves the machine

In this build: nothing. skyhum makes no network requests. Audio is decoded in memory, and reports are written
only where you ask. The `find` log is a local file (`<workdir>/skyhum-find.log`).

When automatic ADS-B download is added (`--adsb auto`, `skyhum adsb fetch`), it will download day files from the
ADS-B archive. Those requests reveal the date and, through the files chosen, the rough area you are interested in.
The audio itself will never be uploaded.

## ADS-B track files

skyhum reads one format: the per-day `tracks.npz` written by the upstream afg pipeline
(`afg adsb-ingest`, [AlanBacker/audio-flyover-geolocation](https://github.com/AlanBacker/audio-flyover-geolocation)), with SI units and altitude above mean sea level. The layout is documented in
`src/skyhum/io/adsb.py`. The synthetic files from `skyhum selftest --write-fixtures` use the same layout.

To build one from public data with the upstream tool:

```
git clone https://github.com/AlanBacker/audio-flyover-geolocation afg-upstream
AFG="python afg-upstream/skills/en/audio-flyover-geolocation/scripts/afg_cli.py"
$AFG adsb-ingest --src traces_dir/ --start 2026-09-08T13:00Z --end 2026-09-08T16:00Z \
    --bbox 33.6,34.5,-118.8,-117.6 --out day.npz
```
`traces_dir/` holds readsb trace files for the day. Sources:

| Source | What | Licence |
|---|---|---|
| [adsb.lol](https://adsb.lol) `globe_history` daily releases | worldwide, one release per day | ODbL 1.0 |
| Your own readsb / tar1090 receiver | local coverage | yours |
| OpenSky Network | research account needed | OpenSky terms |
| ADS-B Exchange free samples | first day of each month only | ADS-B Exchange terms |

Cut the file to the area you need, padded by about half a degree, and to the recording plus at least an hour on
each side (the decoys need it). A whole global day is large; keep only what you need.

**ODbL.** adsb.lol data is under the Open Database Licence. If you publish a database derived from it (a track
file, a list of passes), attribute adsb.lol and publish it under ODbL too. Results such as a verdict are produced
works and only need attribution.

## Recordings

The repository holds no audio, and the benchmark catalogues hold only links and metadata. Each recording keeps its
own licence on its original host:

- xeno-canto: per recording; most are CC BY-NC-SA, a sizeable share CC BY-NC-ND (no derivatives), a few CC BY or
  CC0. Labels derived from ND recordings are not shipped.
- Freesound: per sound; CC0, CC BY or CC BY-NC. Downloads need your own API key.
- Australian Acoustic Observatory (A2O): CC BY 4.0.
- Webcams and public events (YouTube): the platform's standard licence. Only video ids and times are listed.

## Detector weights

None in this build. The classic ridge detector needs no weights. When the learned detector ships, its weights will
be downloaded on first use and checked against a pinned sha256; they will never be stored in git.
