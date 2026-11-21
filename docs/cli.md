# CLI reference

Generated from the argparse definitions by `scripts/gen_cli_docs.py`. Do not edit by hand.

```text
usage: skyhum [-h] [--version] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
              COMMAND ...

Test whether audio carries an aircraft-Doppler location clue. Verify a claimed place, audit your own recordings, scrub the clue. Runs locally.

positional arguments:
  COMMAND
    selftest                synthetic plane + synthetic track: checks the install and prints PASS or
                            FAIL
    doctor                  check Python packages, ffmpeg, GPU, ADS-B cache and detector weights
    regions                 named regions accepted by --claim/--where/--area
    detect                  list aircraft-like tonal passes in the audio (no location involved)
    verify                  is the audio consistent with a claimed place and time?
    audit                   you say where you were: how much did the audio give away?
    scrub                   remove the clue: low-band gate in pauses, or tone scrubbing
    find                    rank places inside a supplied area (research and validation)
    bench                   public benchmark: sources (links and metadata), fetch, scoring
    country                 country search: static tables and the scorers (score, prior, claims) of
                            search outputs
    adsb                    download and index ADS-B history for an area and dates

options:
  -h, --help                show this help message and exit
  --version                 show program's version number and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

## skyhum selftest

```text
usage: skyhum selftest [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                       [--seed SEED] [--quick] [--write-fixtures DIR]

Synthetic propeller passes and synthetic ADS-B tracks. Checks that the detector finds the passes, that verify (reference matcher, then the r5b locator) supports the true place and rejects a displaced place and a wrong clock, and that the tone scrubber works. CPU only: seconds with --quick, minutes without.

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
  --seed SEED               seed of the synthetic world (default: 7)
  --quick                   skip the r5b locator checks (they take minutes on a CPU)
  --write-fixtures DIR      also write the synthetic demo files (clip.wav, call.wav, quiet.wav,
                            tracks.npz, area.geojson) to DIR and exit

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

## skyhum doctor

```text
usage: skyhum doctor [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]

check Python packages, ffmpeg, GPU, ADS-B cache and detector weights

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

## skyhum regions

```text
usage: skyhum regions [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v] {list}

named regions accepted by --claim/--where/--area

positional arguments:
  {list}                    list the bundled named regions

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

## skyhum detect

```text
usage: skyhum detect [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                     [--channel CHANNEL] [--start TIME] [--tz ZONE] [--start-from-file]
                     [--start-tol DUR] [--duration DUR] [--adsb PATH|auto] [--out DIR]
                     [--report FILE] [--ground-alt M] [--spacing KM] [--seed SEED] [--png FILE]
                     AUDIO

List aircraft-like tonal passes found in the audio, with an optional spectrogram PNG. Uses the classic ridge detector in this build (the learned detector is pending).

positional arguments:
  AUDIO                     wav, flac, ogg; mp3, m4a, webm need ffmpeg

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
  --channel CHANNEL         channel to use, 0-based (default: 0)
  --png FILE                write the spectrogram with detections to FILE (needs matplotlib)

clock:
  --start TIME              when the recording began: UTC ISO-8601 (2026-09-08T14:03:00Z) or epoch
                            seconds
  --tz ZONE                 time zone for a --start without Z or offset, e.g. Europe/London
  --start-from-file         take the start from container metadata or file mtime (prints a warning)
  --start-tol DUR           clock uncertainty: 5s, 30s (default), 30m; 2h and 'date' are not in this
                            build's CPU matcher. Wider means more chance matches
  --duration DUR            use only the first DUR of the audio, e.g. 10m

data and output:
  --adsb PATH|auto          ADS-B tracks.npz day file (afg layout); 'auto' download is not in this
                            build
  --out DIR                 directory for extra outputs (spectrogram PNG)
  --report FILE             write a report: FILE.json or a self-contained FILE.html

advanced:
  --ground-alt M            ground altitude in metres (no terrain model in this build; default 0)
  --spacing KM              grid spacing (default: automatic)
  --seed SEED               seed for decoy clocks (default: 0)

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

## skyhum verify

```text
usage: skyhum verify [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                     [--channel CHANNEL] [--start TIME] [--tz ZONE] [--start-from-file]
                     [--start-tol DUR] [--duration DUR] [--adsb PATH|auto] [--out DIR]
                     [--report FILE] [--ground-alt M] [--spacing KM] [--seed SEED] --claim PLACE
                     [--versus PLACE] [--operating-point {default,strict}]
                     [--matcher {auto,r5b,reference}] [--decoys N] [--fast] [--dem-dir DIR]
                     AUDIO

Score the claimed place at the stated clock against decoys (the same place at wrong clocks and other days, and optionally a rival place). SUPPORTED only if the claim beats every decoy. UNCORROBORATED is not evidence that the claim is false.

positional arguments:
  AUDIO                       wav, flac, ogg; mp3, m4a, webm need ffmpeg

options:
  -h, --help                  show this help message and exit
  --device {auto,cpu,cuda}    device for the r5b locator: auto, cpu or cuda (default: auto; cpu
                              never touches a GPU)
  --workdir DIR               working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                      print one JSON document to stdout
  -q, --quiet                 print only the result line
  -v, --verbose               print progress to stderr
  --channel CHANNEL           channel to use, 0-based (default: 0)
  --claim PLACE               a place: a named region ("Los Angeles basin", see `skyhum regions
                              list`), a circle "lat,lon,r=5km", a box "bbox:west,south,east,north",
                              or "@file.geojson"
  --versus PLACE              a rival place, scored as one more decoy
  --operating-point {default,strict}
                              default: beats every decoy; strict: beats them by a fixed margin (not
                              calibrated)

clock:
  --start TIME                when the recording began: UTC ISO-8601 (2026-09-08T14:03:00Z) or epoch
                              seconds (required unless --start-from-file)
  --tz ZONE                   time zone for a --start without Z or offset, e.g. Europe/London
  --start-from-file           take the start from container metadata or file mtime (prints a
                              warning)
  --start-tol DUR             clock uncertainty: 5s, 30s (default), 30m; 2h and 'date' are not in
                              this build's CPU matcher. Wider means more chance matches
  --duration DUR              use only the first DUR of the audio, e.g. 10m

data and output:
  --adsb PATH|auto            ADS-B tracks.npz day file (afg layout); 'auto' download is not in this
                              build
  --out DIR                   directory for extra outputs (spectrogram PNG)
  --report FILE               write a report: FILE.json or a self-contained FILE.html

advanced:
  --ground-alt M              ground altitude in metres (no terrain model in this build; default 0)
  --spacing KM                grid spacing (default: automatic)
  --seed SEED                 seed for decoy clocks (default: 0)

matcher:
  --matcher {auto,r5b,reference}
                              r5b: the round-5b locator (test partial; needs PyTorch and its fitted
                              tables, slow on CPU); reference: the simplified numpy matcher (fast,
                              not benchmarked); auto (default): r5b when available, otherwise
                              reference with a notice
  --decoys N                  decoy clocks to try (default: 20 with r5b, 200 with reference)
  --fast                      r5b: fewer calibration passes and refinements; for checking the
                              install, not for results
  --dem-dir DIR               r5b: directory of GeoTIFF terrain tiles (otherwise flat ground at
                              --ground-alt); nothing is downloaded

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

## skyhum audit

```text
usage: skyhum audit [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                    [--channel CHANNEL] [--start TIME] [--tz ZONE] [--start-from-file]
                    [--start-tol DUR] [--duration DUR] [--adsb PATH|auto] [--out DIR]
                    [--report FILE] [--ground-alt M] [--spacing KM] [--seed SEED] --where PLACE
                    [--scrub OUT] [--method {gate,tones}] [--matcher {auto,r5b,reference}]
                    [--decoys N] [--fast] [--dem-dir DIR]
                    AUDIO

You supply the truth. Reports detected passes, ADS-B passes in range, whether verify accepts the true place, whether a place 15 km away also passes, and fixed-rule advice. With --scrub, repeats the audit on scrubbed audio.

positional arguments:
  AUDIO                       wav, flac, ogg; mp3, m4a, webm need ffmpeg

options:
  -h, --help                  show this help message and exit
  --device {auto,cpu,cuda}    device for the r5b locator: auto, cpu or cuda (default: auto; cpu
                              never touches a GPU)
  --workdir DIR               working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                      print one JSON document to stdout
  -q, --quiet                 print only the result line
  -v, --verbose               print progress to stderr
  --channel CHANNEL           channel to use, 0-based (default: 0)
  --where PLACE               where you were. a place: a named region ("Los Angeles basin", see
                              `skyhum regions list`), a circle "lat,lon,r=5km", a box
                              "bbox:west,south,east,north", or "@file.geojson"
  --scrub OUT                 also scrub the audio to OUT and audit the result
  --method {gate,tones}       scrub method for --scrub

clock:
  --start TIME                when the recording began: UTC ISO-8601 (2026-09-08T14:03:00Z) or epoch
                              seconds (required unless --start-from-file)
  --tz ZONE                   time zone for a --start without Z or offset, e.g. Europe/London
  --start-from-file           take the start from container metadata or file mtime (prints a
                              warning)
  --start-tol DUR             clock uncertainty: 5s, 30s (default), 30m; 2h and 'date' are not in
                              this build's CPU matcher. Wider means more chance matches
  --duration DUR              use only the first DUR of the audio, e.g. 10m

data and output:
  --adsb PATH|auto            ADS-B tracks.npz day file (afg layout); 'auto' download is not in this
                              build
  --out DIR                   directory for extra outputs (spectrogram PNG)
  --report FILE               write a report: FILE.json or a self-contained FILE.html

advanced:
  --ground-alt M              ground altitude in metres (no terrain model in this build; default 0)
  --spacing KM                grid spacing (default: automatic)
  --seed SEED                 seed for decoy clocks (default: 0)

matcher:
  --matcher {auto,r5b,reference}
                              r5b: the round-5b locator (test partial; needs PyTorch and its fitted
                              tables, slow on CPU); reference: the simplified numpy matcher (fast,
                              not benchmarked); auto (default): r5b when available, otherwise
                              reference with a notice
  --decoys N                  decoy clocks to try (default: 20 with r5b, 200 with reference)
  --fast                      r5b: fewer calibration passes and refinements; for checking the
                              install, not for results
  --dem-dir DIR               r5b: directory of GeoTIFF terrain tiles (otherwise flat ground at
                              --ground-alt); nothing is downloaded

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

## skyhum scrub

```text
usage: skyhum scrub [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                    [--method {gate,tones}] [--cutoff HZ] [--channel CHANNEL] [--seed SEED]
                    IN OUT

gate: replace the band below --cutoff with comfort noise while nobody speaks (WebRTC VAD). tones: track long tonal lines (aircraft, machinery) and replace them with noise at the local floor, under speech too. Tested in offline simulation only; real platform pipelines may behave differently.

positional arguments:
  IN                        input audio
  OUT                       output audio (.wav or .flac; others need ffmpeg)

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
  --method {gate,tones}     default: gate
  --cutoff HZ               gate cutoff in Hz (default: 400)
  --channel CHANNEL         input channel, 0-based
  --seed SEED               comfort-noise seed

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

## skyhum find

```text
usage: skyhum find [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                   [--channel CHANNEL] [--start TIME] [--tz ZONE] [--start-from-file]
                   [--start-tol DUR] [--duration DUR] [--adsb PATH|auto] [--out DIR] [--report FILE]
                   [--ground-alt M] [--spacing KM] [--seed SEED] --area PLACE [--top TOP]
                   [--sessions A,B] [--session-starts T1,T2] [--single-event]
                   [--matcher {auto,r5b,reference}] [--decoys N] [--fast] [--dem-dir DIR]
                   AUDIO

Rank candidate places inside an area you supply. A point estimate is printed only above the strict threshold. Confidence is not calibrated: about 2 in 3 confident answers within 2 km on a 7-cluster test; the strict threshold is an uncalibrated margin. For research, validation and defence; not for locating private individuals.

positional arguments:
  AUDIO                       wav, flac, ogg; mp3, m4a, webm need ffmpeg

options:
  -h, --help                  show this help message and exit
  --device {auto,cpu,cuda}    device for the r5b locator: auto, cpu or cuda (default: auto; cpu
                              never touches a GPU)
  --workdir DIR               working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                      print one JSON document to stdout
  -q, --quiet                 print only the result line
  -v, --verbose               print progress to stderr
  --channel CHANNEL           channel to use, 0-based (default: 0)
  --area PLACE                search area. a place: a named region ("Los Angeles basin", see `skyhum
                              regions list`), a circle "lat,lon,r=5km", a box
                              "bbox:west,south,east,north", or "@file.geojson"
  --top TOP                   ranked tiles to show (default: 5)
  --sessions A,B              more recordings you state come from the same spot
  --session-starts T1,T2      start times of --sessions, same order
  --single-event              r5b: allow answers from one pass (clock within 5 s); prints the mirror
                              point too

clock:
  --start TIME                when the recording began: UTC ISO-8601 (2026-09-08T14:03:00Z) or epoch
                              seconds (required unless --start-from-file)
  --tz ZONE                   time zone for a --start without Z or offset, e.g. Europe/London
  --start-from-file           take the start from container metadata or file mtime (prints a
                              warning)
  --start-tol DUR             clock uncertainty: 5s, 30s (default), 30m; 2h and 'date' are not in
                              this build's CPU matcher. Wider means more chance matches
  --duration DUR              use only the first DUR of the audio, e.g. 10m

data and output:
  --adsb PATH|auto            ADS-B tracks.npz day file (afg layout); 'auto' download is not in this
                              build
  --out DIR                   directory for extra outputs (spectrogram PNG)
  --report FILE               write a report: FILE.json or a self-contained FILE.html

advanced:
  --ground-alt M              ground altitude in metres (no terrain model in this build; default 0)
  --spacing KM                grid spacing (default: automatic)
  --seed SEED                 seed for decoy clocks (default: 0)

matcher:
  --matcher {auto,r5b,reference}
                              r5b: the round-5b locator (test partial; needs PyTorch and its fitted
                              tables, slow on CPU); reference: the simplified numpy matcher (fast,
                              not benchmarked); auto (default): r5b when available, otherwise
                              reference with a notice
  --decoys N                  decoy clocks to try (default: 10 with r5b, 200 with reference)
  --fast                      r5b: fewer calibration passes and refinements; for checking the
                              install, not for results
  --dem-dir DIR               r5b: directory of GeoTIFF terrain tiles (otherwise flat ground at
                              --ground-alt); nothing is downloaded

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

## skyhum bench

```text
usage: skyhum bench [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v] ACTION ...

public benchmark: sources (links and metadata), fetch, scoring

positional arguments:
  ACTION
    list                    list sources with licences
    fetch                   download recordings from the original host (not in this build)
    run                     rerun or rescore a family of published rows (REPRODUCING.md)

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

### skyhum bench list

```text
usage: skyhum bench list [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
```

### skyhum bench fetch

```text
usage: skyhum bench fetch [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                          --source SOURCE [--n N]

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
  --source SOURCE
  --n N
```

### skyhum bench run

```text
usage: skyhum bench run [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                        [--reproduce FAMILY] [--set {a2o,webcam}] [--jobs JOBS] [--manifest FILE]
                        [--audio-dir DIR] [--hints-dir DIR] [--spec-dir DIR] [--outputs DIR]
                        [--truth FILE] [--cases FILE] [--static DIR] [--channel CHANNEL]
                        [--calls-runs DIR] [--calls-truth FILE] [--tag TAG] [--ref-tag REF_TAG]
                        [--decoy] [--adsb-dir DIR] [--country-root DIR]

options:
  -h, --help                 show this help message and exit
  --device {auto,cpu,cuda}   device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                             touches a GPU)
  --workdir DIR              working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                     print one JSON document to stdout
  -q, --quiet                print only the result line
  -v, --verbose              print progress to stderr
  --reproduce FAMILY         row family to rerun or rescore: country, prior, verify, box, single-
                             event (REPRODUCING.md)
  --set {a2o,webcam}         benchmark set (default: a2o; webcam needs the webcam feature release)
  --jobs JOBS                searches in parallel (default: 1)
  --manifest FILE            a manifest other than the shipped one (feature releases)
  --audio-dir DIR            rerun: recordings named <item id>.<ext>
  --hints-dir DIR            rerun: detector-v3 detections <item id>.json
  --spec-dir DIR             rerun: spectrograms <item id>.npz instead of audio
  --outputs DIR, --runs DIR  rescore: existing locator outputs (yours or the published ones) instead
                             of searching
  --truth FILE               rescore (country families): truth.json of --outputs
  --cases FILE               rescore (box families): cases JSON of --outputs (default: the shipped
                             A2O manifest)
  --static DIR               static tables (default: WORKDIR/country/static)
  --channel CHANNEL          country score channel (default: s2, tonal)
  --calls-runs DIR           verify: outputs of the same search on call audio
  --calls-truth FILE         verify: truth of --calls-runs
  --tag TAG                  box families: tag of the runs (default: r5b_test / r5b_se)
  --ref-tag REF_TAG          box: a second tag scored on the same cases (e.g. the round-5a locator)
  --decoy                    box: use the clock-decoy runs (<runs>/<tag>__decoy)
  --adsb-dir DIR             box rerun: low-altitude ADS-B day files
  --country-root DIR         country work directory (default: WORKDIR/country)
```

## skyhum country

```text
usage: skyhum country [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v] ACTION ...

positional arguments:
  ACTION
    static                  download the public boundary / population / terrain files and build the
                            region rasters
    score                   score search outputs: region top-1, tiers, decoy confidence, multi-
                            session
    prior                   fuse with population / UTC-offset priors; value of the sound given a
                            prior
    claims                  verify claimed places (true and false claims) from search outputs

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

### skyhum country static

```text
usage: skyhum country static [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                             [--country-root DIR] [--country {US,AU,both}] [--raw DIR]

download the public boundary / population / terrain files and build the region rasters

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
  --country-root DIR        country work directory (default: WORKDIR/country)
  --country {US,AU,both}
  --raw DIR                 use already downloaded public files from DIR
```

### skyhum country score

```text
usage: skyhum country score [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                            [--country-root DIR] --truth TRUTH [--static DIR] --out PATH
                            [--channel CHANNEL]
                            runs

score search outputs: region top-1, tiers, decoy confidence, multi-session

positional arguments:
  runs                      directory of search outputs (*.json)

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
  --country-root DIR        country work directory (default: WORKDIR/country)
  --truth TRUTH             truth.json the runs were staged with
  --static DIR              static tables (default: ROOT/static)
  --out PATH                output prefix (score) or directory
  --channel CHANNEL         score channel: s2 tonal (default), l_s2, c1_s2
```

### skyhum country prior

```text
usage: skyhum country prior [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                            [--country-root DIR] --truth TRUTH [--static DIR] --out PATH
                            runs

fuse with population / UTC-offset priors; value of the sound given a prior

positional arguments:
  runs                      directory of search outputs (*.json)

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
  --country-root DIR        country work directory (default: WORKDIR/country)
  --truth TRUTH             truth.json the runs were staged with
  --static DIR              static tables (default: ROOT/static)
  --out PATH                output prefix (score) or directory
```

### skyhum country claims

```text
usage: skyhum country claims [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v]
                             [--country-root DIR] --truth TRUTH [--static DIR] --out PATH
                             [--channel CHANNEL] [--calls-runs DIR] [--calls-truth FILE]
                             runs

verify claimed places (true and false claims) from search outputs

positional arguments:
  runs                      directory of search outputs (*.json)

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
  --country-root DIR        country work directory (default: WORKDIR/country)
  --truth TRUTH             truth.json the runs were staged with
  --static DIR              static tables (default: ROOT/static)
  --out PATH                output prefix (score) or directory
  --channel CHANNEL         score channel: s2 tonal (default), l_s2, c1_s2
  --calls-runs DIR          outputs of the same search on call audio
  --calls-truth FILE
```

## skyhum adsb

```text
usage: skyhum adsb [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v] ACTION ...

download and index ADS-B history for an area and dates

positional arguments:
  ACTION
    fetch                   download ADS-B days (not in this build)

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr

exit codes:
  0   success (verify: SUPPORTED)        10  UNCORROBORATED (not proof the claim is false)
  11  INCONCLUSIVE (too little to test)  2   usage error, or not implemented in this build
  3   missing data (ADS-B, input file)
  1   selftest failed or internal error

Audio never leaves this machine. See docs/ for concepts, limits and responsible use.
```

### skyhum adsb fetch

```text
usage: skyhum adsb fetch [-h] [--device {auto,cpu,cuda}] [--workdir DIR] [--json] [-q] [-v] --area
                         PLACE --date YYYY-MM-DD

options:
  -h, --help                show this help message and exit
  --device {auto,cpu,cuda}  device for the r5b locator: auto, cpu or cuda (default: auto; cpu never
                            touches a GPU)
  --workdir DIR             working directory (default: $SKYHUM_WORKDIR or ~/.local/share/skyhum)
  --json                    print one JSON document to stdout
  -q, --quiet               print only the result line
  -v, --verbose             print progress to stderr
  --area PLACE
  --date YYYY-MM-DD
```
