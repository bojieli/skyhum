# Reproducing the published numbers

**Source of every number.** The numbers to compare against are those of the article, published at
https://x.com/bojie_li/status/2124169630596137359. This file maps each published row to the article section it
appears in, the command that recomputes it, its inputs, and whether a third party can run it today. Row ids (A2, B3,
V, ...) are the ids of the project's findings table; section names are the article's headings.

## How to read the tables

* **Command.** `skyhum bench run --reproduce FAMILY` rescores locator outputs when you pass `--outputs DIR` (and
  `--truth FILE` or `--cases FILE`), or reruns the search from the benchmark items when you do not (`--audio-dir`,
  `--hints-dir`). Rescoring is CPU-only and takes minutes. See `skyhum bench run --help` and docs/locate.md.
* **Inputs.**
  * *A2O manifests* ship in `src/skyhum/bench/manifests/` (links and metadata only): `a2o_country_v1.json` (139
    one-hour A2O sensor recordings, 13 same-sensor groups, the clock given to the search, decoy days and labelled pass
    times) and `a2o_box_v1.json` (76 A2O hours of the held-out box test). Download each recording from its `url`
    (Australian Acoustic Observatory, CC BY 4.0).
  * *Saved outputs bundle*: the research locator outputs for the A2O and public-webcam units, with their truth files
    (published separately; it holds no event or speech unit). It lets you rescore without a GPU.
  * *ADS-B*: adsb.lol daily releases (ODbL 1.0), fetched by `skyhum adsb fetch --country`.
  * *Static tables*: `skyhum country static` downloads the US Census CBSA 2023 / county files, the ABS SA4 2021
    boundaries and AWS terrain tiles and builds the region rasters (bit-identical to the research build).
  * *Detector hints*: detector-v3 detections per recording. A rerun needs them; detector inference is being added to
    the package separately.
* **Third party today.**
  * *Yes (rescore)*: reproducible from the saved outputs bundle, CPU, minutes.
  * *Yes (rerun)*: also rerunnable from the public recordings (`--audio-dir`) with ADS-B fetch and the search in
    this build; detector-v3 detections are needed per recording (`--hints-dir`) until detector inference ships.
  * *Needs webcam features*: webcam audio is not redistributable. These rows need the webcam derived-feature release
    (public airport and city webcam hours); the commands take its manifest with `--manifest FILE` and its
    spectrograms with `--spec-dir`.
  * *Not reproducible by design*: event and speech livestream items (rallies, protests, amateur radio and similar
    streams that could identify or locate a private individual) are left out of every public manifest and bundle.
  * *Needs live call accounts*: the rows that went through real Google Meet, Telegram, Zoom or FaceTime calls.
  * *Not ported*: the measurement came from a research workstream whose code is not in this package yet.
* **Runtime** is measured from the research run logs (sum of per-group search time on one data-centre GPU) unless
  marked *est.*. Rescoring takes CPU minutes. CPU searches are much slower than GPU ones; docs/locate.md gives the
  measured CPU/GPU ratio for one group.

## Rows that include event and speech items (not third-party reproducible)

Event and speech livestream items are not published. Third parties therefore cannot reproduce:

| what | units left out |
|---|---|
| events: labeller audibility (C0) | 95 + 49 recordings |
| events: country search and claims (C2, C3, C3n, C6, C7; events rows of section V) | 138 one-hour units |
| events: metro-box real-world runs (C4, C5) | 100 + 100 recordings |
| speech: labeller audibility (D0) | 92 recordings |
| speech: country search, metro box and claims (D2-D6, D3n; speech rows of section V) | 79 one-hour units (D4: 92 recordings) |
| the US one-hour summary (15/327 sound only; 89/327 with the UTC-offset guess; the baselines 50/327 and 87/327) | 327 units, of which 138 events and 79 speech |
| the 634-recording real-world confident-fix set | events v0 100, v1 100, speech 92 of 634 |

The webcam part of the summary (7/110) is reproducible once the webcam feature release exists.

## Article section: "Far hum hints at the region; a near pass can pin a spot" and the headline figure

| row | what | command | inputs | third party today | runtime | compare with |
|---|---|---|---|---|---|---|
| A2 | A2O, country search, ±30 s, N = 1 | `skyhum bench run --reproduce country --outputs O --truth T` (60 min, ±30 s row of the a2o table) | A2O manifest; bundle | Yes (rescore) | rerun: 8.2 GPU-h for all A2O groups (every window, clock and N) | 39/139; tiers 1 / 7 / 17 / 20 / 33 / 33; 39 / 43 / 49; own 25 / 9; pooled 20 / 0 |
| A3 | A2O, country search, ±5 s, N = 1 | same, 60 min ±5 s row | same | Yes (rescore) | same run as A2 | 37/139; 1 / 7 / 21 / 22 / 31 / 32; 35 / 39 / 46; own 22 / 8; pooled 19 / 2; margin > 5: 16 / 0 |
| A3t | A2O, + UTC offset | `--reproduce prior` (a2o, "acoustic + UTC offset") | same + timezonefinder | Yes (rescore) | CPU minutes | 50/139; 1 / 8 / 29 / 30 / 41 / 42 |
| A20 | real-case figures | not ported (figure scripts) | A2O recordings | Not ported | – | as published |

## Article section: "One hour of hum lost to guessing New York; more hours helped"

| row | what | command | inputs | third party today | runtime | compare with |
|---|---|---|---|---|---|---|
| summary | US one hour, sound only; baselines | `--reproduce country --set webcam` (webcam part only) | webcam features | Not reproducible by design (events, speech); webcam part: needs webcam features | 6.8 GPU-h (webcam groups) | 15/327 (webcams 7/110); New York 50/327; offset guess 87/327 |
| summary | sound + UTC-offset guess | `--reproduce prior` | same | Not reproducible by design (webcam part: needs webcam features) | CPU minutes | 89/327 |
| A4 | A2O, N = 2 | `--reproduce country` (multi-session a2o table) | A2O manifest; bundle | Yes (rescore) | in A2 | 32/68; 2 / 9 / 24 / 26 / 32 / 32; own 25 / 2; pooled 19 / 0 |
| A5 | A2O, N = 4 | same | same | Yes (rescore) | in A2 | 23/32; 1 / 8 / 17 / 19 / 23 / 23; own 20 / 1; pooled 19 / 0 |
| A6 | A2O, N = 8 | same | same | Yes (rescore) | in A2 | 13/14; 1 / 3 / 10 / 12 / 13 / 13; own 13 / 0; pooled 11 / 0 |
| A6w | A2O, N = 4, every session ≤ 1 heard pass | same ("accumulation of weak sessions" table) | same | Yes (rescore) | in A2 | 7/10; own 6 / 0 |
| A18 | tonal / loud / combined channels | `--reproduce country --channel s2 / l_s2 / c1_s2` on the loud-channel outputs | bundle (loud-channel run) | Yes (rescore) for the stored channels; rerun needs the loud channel (ported separately) | rerun: 7.4 GPU-h (A2O), 47.9 GPU-h (webcams) | A2O N=1 37 / 8 / 39 of 139; N=4 23 / 7 / 23 of 32; N=8 13 / 2 / 13 of 14; webcams N=1 8 / 2 / 7 of 110, N=8 3 / 1 / 3 of 8 |
| A18d | loud channel weight choice (dev) | not ported (dev staging) | dev A2O hours | Not ported | – | as published |
| A18v | verifier on the combined score | pending in the source | – | – | – | pending |
| B2 | webcams, ±30 s, N = 1 | `--reproduce country --set webcam --manifest M` (or `--outputs` with the bundle) | webcam features; bundle | Yes (rescore); rerun needs webcam features | in summary | 6/110; 4 / 6 / 6 / 6 / 6 / 6; own 4 / 5; pooled 4 / 3 |
| B3 | webcams, ±5 s, N = 1 | same | same | same | – | 7/110; 4 / 6 / 6 / 6 / 6 / 6; 6 / 6 / 7; own 4 / 4; pooled 5 / 2 |
| B3p | webcams + population and offset, gated | `--reproduce prior` | same | same | CPU minutes | gated 46/110; guess alone 46/110 |
| B4 | webcams, N = 2 | `--reproduce country` | same | same | – | 7/52; own 6 / 3; pooled 2 / 2 |
| B5 | webcams, N = 4 | same | same | same | – | 6/24; gated 13/24; guess 10/24 |
| B6 | webcams, N = 8 | same | same | same | – | 3/8; own 3 / 0 |
| B6w | webcams, N = 4, weak sessions | same | same | same | – | 3/18; own 3 / 0 |
| C2, C3, C3n, D2, D3, D3n | events and speech, country search | – | – | Not reproducible by design | – | C3 4/138; D3 4/79 |
| E2-E4, F2-F4 | country search on received call audio | `--reproduce verify --calls-runs ... --calls-truth ...` scores stored call outputs | call recordings (4 webcams x 4 h through Meet / Telegram) | Needs live call accounts (and webcam features) | 9.1 GPU-h | E2 3/16; F2 3/16; E3 2/4; F3 2/4 |

## Article section: "Meetings have the one thing podcasts don't: a clock" (the clock figure)

| row | what | command | inputs | third party today | runtime | compare with |
|---|---|---|---|---|---|---|
| A17 | round-5b locator, jittered 0.25° box, date only | `skyhum bench run --reproduce box --outputs O --cases C --tag r5b_test --ref-tag v3final_v2j` | box manifest (A2O part); bundle (all 43 positives: 38 A2O, 5 webcam) | Yes (rescore); a rerun covers the 38 A2O positives, the 5 webcam positives need webcam features | rerun: 3.6 GPU-h (all three clocks, 93 cases) | ≤ 2 km 22/43; tiers 1 / 4 / 16 / 22 / 26 / 29 |
| A17b | same, ±30 min | same | same | same | in A17 | 23/43; 1 / 6 / 18 / 23 / 30 / 33 |
| A17c | same, ±30 s | same | same | same | in A17 | 31/43; ≤ 1 km 24/43; 1 / 5 / 24 / 31 / 40 / 42 |
| A7-A9 | round-5a locator, same box | same command (the `v3final_v2j` rows) | same | Yes (rescore); a rerun needs the round-5a locator, not in the package | 14.7 GPU-h | 15/43, 19/43, 23/43; tiers as published |
| A1 | ±1° box, date only | not ported | – | Not ported | – | as published |
| B1, C1, D1 | date only, country | – | – | not measured | – | – |

## Article section: "A single detected close pass can land within a kilometre"

| row | what | command | inputs | third party today | runtime | compare with |
|---|---|---|---|---|---|---|
| A17d | single event, two-point answer, 38 hard hours | `--reproduce single-event --outputs O --cases C --tag r5b_se` | box manifest; bundle | Partly: the bundle holds 35 of the 38 hard hours (33 A2O, 2 webcam; the 3 xeno-canto hours are not in it, all three misses), so it gives 24/35 | 0.4 GPU-h; decoys 3.6 GPU-h | 24/38 within 2 km |
| A17f | same, with clock decoys (p ≤ 0.05) | same | same + decoy outputs | Partly: from the bundle, hard 21 confident, 14 / 7 (one confident-wrong xeno-canto hour is not in it); positives 33, 26 / 7 exactly; negatives 1 wrong of 8 answered | in A17d | hard 22 confident, 14 / 8; positives 33, 26 / 7; negatives 1 wrong of 9; all 40/56 |
| A13, A13m, A14, A15 | loud single pass / hour | loud channel (ported separately) | – | Not ported here | – | as published |
| A16, A16c | range terms | not ported (ranging workstream) | – | Not ported | – | as published |
| A19 | upstream comparison | not ported | – | Not ported | – | 7/8 within 1 km |

## Article section: "At our default rule, checking a claimed region was easier than finding it" (section V)

All rows: `skyhum bench run --reproduce verify --outputs O --truth T` (call rows add `--calls-runs`, `--calls-truth`).
CPU minutes; needs `pip install "skyhum[country]"` (timezonefinder).

| row | sample | third party today | compare with |
|---|---|---|---|
| V webcams N = 1 | true claims, default and stringent | Yes (rescore from bundle); rerun needs webcam features | 18/110; stringent 11/110 |
| V webcams N = 2 / 4 / 8 | | same | 12/52, 9/24, 4/8; stringent 9/52, 6/24, 4/8 |
| V A2O N = 1 | | Yes (rescore) | 52/139; stringent 16/139 |
| V A2O N = 2 / 4 / 8 | | Yes (rescore) | 42/68, 24/32, 13/14; stringent 32/68, 21/32, 13/14 |
| V A2O 40 km box | | Yes (rescore) | 49/139; stringent 9/139 |
| V false claims, webcams and A2O | by claim type | Yes (rescore; webcams from bundle) | as published per type (e.g. A2O N = 1 next region 6/139) |
| V events, speech | | Not reproducible by design | 7/138, 7/79 |
| V webcam hours ≥ 2 heard | earlier rule | Yes (rescore, `--reproduce prior`) | 6/11 |
| V through real calls | Meet / Telegram / unprocessed | Needs live call accounts | as published |
| A12, B10 | find vs verify | Yes (rescore) | find 37/139 and 7/110 |

## Value of the sound given a prior (rows A10, A11, B8, B8n, B9)

`--reproduce prior --outputs O --truth T`, CPU minutes. A2O rows: Yes (rescore). Webcam rows: Yes (rescore from the
bundle); rerun needs webcam features. C6 and D5: not reproducible by design.

| row | compare with |
|---|---|
| A10 | 1 / 8 / 25 / 27 / 44 / 45 |
| A11 | 1 / 8 / 26 / 29 / 76 / 124 |
| B8 | 4 / 6 / 11 / 22 / 37 / 54; sound used 18/110 (15 closer, 3 farther) |
| B8n | N = 4: 5 / 6 / 6 / 7 / 9 / 13; N = 8: 3 / 3 / 3 / 4 / 4 / 5 |
| B9 | 4 / 6 / 11 / 13 / 44 / 94 |

## Article section: "What survives a replay through the apps"

Rows E0-E7, F0-F8, G1-G6 used real desktop calls (Google Meet, Telegram Desktop, Zoom, FaceTime): **needs live call
accounts**; the call-test harness is not in the package. Rows I1-I8 (simulated codecs) and H1-H4 (simulated indoor)
come from research workstreams not ported here: **not ported**.

## Article section: "Helicopters: easiest to hear, hardest to detect" and "Modelled odds rose near small airfields"

Rows K1-K11 (urban relabel, detector-urban, models) and J1-J9 (low jets) come from research workstreams not ported
here: **not ported**. J8 and J9 were pending in the source.

## Article section: "What you can do, and what apps could do" (mitigation figure)

| figure value | what | third party today | compare with |
|---|---|---|---|
| do nothing | 53 selected quiet A2O hours, ±30 s, round-3 locator | Not ported (the round-3 locator is not in the package); the A2O hours are public | 32/53 |
| re-encode (AAC 96k) | same | same | 32/53 |
| scrub aircraft tones | same (scrubber ported as `skyhum scrub --method tones`) | same | 17/53 |
| gate the band below 400 Hz in pauses | same (gate ported as `skyhum scrub --method gate`) | same | 4/53 |

## Article section: "How we tested, and what the numbers cover"

Method notes only; no extra rows.
