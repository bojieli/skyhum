# Whole-country search

The research question behind the published country rows: given a recording whose start time is known to a few
seconds, which region of a whole country explains the aircraft passes in it? Regions are US Census CBSAs 2023
(metropolitan and micropolitan areas, else the county) in the contiguous US, and ABS Statistical Areas Level 4 (SA4,
2021) in Australia.

**Base rate.** In the published benchmark the correct US region was ranked first for about 5 % of one-hour
recordings (15/327, sound only, clock known to ±5 s); Australian rural sensors 37/139 hours. **No answer is
validated** as correct for a new recording, and the confidence of the round-5b locator is not calibrated (about 2 in 3 confident answers within 2 km on a 7-cluster test). The country
search is an explicit research command; `verify` and `audit` stay the front door for everyone else. For research,
validation and defence; not for locating private individuals.

## Pipeline

| step | what | command |
|---|---|---|
| static tables | region rasters (30"), DEM, county population, from public files downloaded at run time | `skyhum country static` |
| ADS-B days | adsb.lol daily releases cut to the country's low traffic (≤ 10,000 ft) | `skyhum adsb fetch --country` |
| detection | detector-v3 detections per recording (`hints_v3.json`) | supplied with `--hints` until detector inference ships |
| search | candidates from detections, round-5b per-pass LLR scoring, decoy variants, refinement | `skyhum locate --country`; benchmark: `skyhum country stage` and `skyhum country run` |
| scoring | region top-1 / top-3, distance tiers, own and pooled decoy confidence, multi-session | `skyhum country score` |
| priors | fusion with population and UTC offset; the sound given a prior | `skyhum country prior` |
| claims | verify a claimed region / 40 km box / 5 km neighbourhood | `skyhum country claims` |

The work directory is `$SKYHUM_WORKDIR/country` (override with `--country-root`): `static/`, `adsb/`, `blind/`,
`groups/`, `truth/`, `runs/`. The scorers are the only parts that open `truth/`; a search process installs an audit
hook that refuses any path under it and records every work-directory path it opened, and the scorer refuses outputs
that touched the truth.

## Locate one recording

```
skyhum country static --country US                      # once
skyhum adsb fetch --country US --date 2026-09-08 ...    # the days locate asks for
skyhum locate --country US rec.flac --start 2026-09-08T14:03:00Z --hints rec.dets.json
```

`--hints` is the detector-v3 output for the recording, `[{"t": seconds, "f": Hz, "p": probability}, ...]`. Without
detections the search proposes no candidate place. `locate` prints the ranked regions with their best points, whether
the top region beats every decoy (8 clock shifts and, by default, the same time 7 days before and after: those days
need their ADS-B files too), the base rate and the limits. `--sessions`, `--session-starts` and `--session-hints` add
recordings you state come from the same place; they are searched jointly. `--report FILE` writes the full output.

On a GPU a one-hour recording takes minutes; on a CPU, hours. Measured on one saved research group (one 1.8 h
recording, one detection): the whole group (calibration and 11 traffic variants) took 21.5 s of search time on a data-
centre GPU; on a shared, heavily loaded 32-core CPU at the lowest priority the calibration took 4.0 h and the real
variant another 4.0 h. That CPU run gave the same ranked regions and best points as the GPU output in all 12 ranked
lists, with scores equal to their 3-decimal rounding. Rescoring saved outputs (`skyhum bench run --reproduce ...
--outputs`) is the practical CPU path.

## ADS-B days

`skyhum adsb fetch --country US|AU|both --date D [--date D ...]` downloads the whole adsb.lol global day (a 2023 day is
about 0.9 GB, recent days several GB in split parts) into `adsb/raw/`, resuming an interrupted download, cuts it to the
country box below 10,000 ft (prop, rotor and unknown aircraft; jets to a side file) and deletes the archive. The cut
files are a few MB to tens of MB per country-day. They are derived from adsb.lol and stay under ODbL 1.0. Aircraft
classes use the ICAO Doc 8643 type table from the tar1090-db project, downloaded on first use.

## Benchmark runs

```
skyhum country stage src/skyhum/bench/manifests/a2o_country_v1.json --audio-dir A --hints-dir H
skyhum adsb fetch --country AU --date ...          # every day the stage step lists
skyhum country run main --jobs 2                   # resumable; one process per group
skyhum country score WORKDIR/country/runs/main --truth WORKDIR/country/truth/truth.json --out main
```

or in one step `skyhum bench run --reproduce country --audio-dir A --hints-dir H --jobs 2`.

## Static tables

`skyhum country static [--country US|AU|both] [--raw DIR]` downloads (about 50 MB plus ~2,000 terrain tiles):

* US Census cartographic boundary files `cb_2023_us_cbsa_500k.zip`, `cb_2023_us_county_500k.zip`, the CBSA
  population estimates `cbsa-est2023-alldata.csv` and the 2020 county centres of population;
* ABS `SA4_2021_AUST_SHP_GDA2020.zip` (CC BY 4.0);
* AWS Terrain Tiles (terrarium, zoom 8).

Built from the same files, the tables are bit-identical to the research build. The Australian population (ABS ERP
2021 by SA4, CC BY 4.0) ships with the package. Needs `pip install "skyhum[country]"`.

## Scorers

```
skyhum country score  RUNS --truth truth.json --out OUT_PREFIX [--channel s2|l_s2|c1_s2]
skyhum country prior  RUNS --truth truth.json --out OUT_DIR
skyhum country claims RUNS --truth truth.json --out OUT_DIR [--calls-runs DIR --calls-truth FILE]
```

`RUNS` holds one search output per group. The truth file maps each session to its set, site, coordinates, clock,
duration, windows and labelled pass times. On the saved research outputs these scorers reproduce the published
tables exactly (REPRODUCING.md).

Definitions: *top-1* = the true region ranked first (no answer counts as a miss). *Confident (own decoys)* = the real
top score beats every decoy of the same unit (8 clock shifts of ±10 to ±25 min and 2 other days). *Confident
(pooled)* = pooled p ≤ 0.01 against every decoy top of the same set, window, clock and N. A claim is accepted at the
default rule when its area's best saved cell beats the same area in every decoy variant and scores above 0.

## Limits

* The saved outputs keep only the best cell of each of the top 40 regions, so in-prior and claim numbers are lower
  bounds on what a search inside the claimed area would find; their false-accept rates can move either way.
* Tested on outdoor public recordings; ordinary indoor or phone callers were not measured.
* The search's clock decoys and other-day decoys are the confidence null; with K = 10 decoys the smallest own-decoy
  p is 1/11, a weak bar.
