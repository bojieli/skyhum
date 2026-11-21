# Whole-country search

The research question behind the published country rows: given a recording whose start time is known to a few
seconds, which region of a whole country explains the aircraft passes in it? Regions are US Census CBSAs 2023
(metropolitan and micropolitan areas, else the county) in the contiguous US, and ABS Statistical Areas Level 4 (SA4,
2021) in Australia.

**Base rate.** In the published benchmark the correct US region was ranked first for about 5 % of one-hour
recordings (15/327, sound only, clock known to ±5 s); Australian rural sensors 37/139 hours. **No answer is
validated** as correct for a new recording, and the confidence of the round-5b locator is not validated. The country
search is an explicit research command; `verify` and `audit` stay the front door for everyone else. For research,
validation and defence; not for locating private individuals.

## Pipeline

| step | what | command |
|---|---|---|
| static tables | region rasters (30"), DEM, county population, from public files downloaded at run time | `skyhum country static` |
| ADS-B days | adsb.lol daily releases cut to the country's low traffic (≤ 10,000 ft) | `skyhum adsb fetch --country` (later build) |
| detection | detector-v3 detections per recording (`hints_v3.json`) | supplied with `--hints` until detector inference ships |
| search | candidates from detections, round-5b per-pass LLR scoring, decoy variants, refinement | `skyhum locate --country` (later build) |
| scoring | region top-1 / top-3, distance tiers, own and pooled decoy confidence, multi-session | `skyhum country score` |
| priors | fusion with population and UTC offset; the sound given a prior | `skyhum country prior` |
| claims | verify a claimed region / 40 km box / 5 km neighbourhood | `skyhum country claims` |

The work directory is `$SKYHUM_WORKDIR/country` (override with `--country-root`): `static/`, `adsb/`, `blind/`,
`groups/`, `truth/`, `runs/`. The scorers are the only parts that open `truth/`; a search process installs an audit
hook that refuses any path under it and records every work-directory path it opened, and the scorer refuses outputs
that touched the truth.

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
