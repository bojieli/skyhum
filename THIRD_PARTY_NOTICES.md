# Third-party notices

## Code

**afg (audio-flyover-geolocation)** by AlanBacker, https://github.com/AlanBacker/audio-flyover-geolocation,
commit 4c2465b. MIT License, Copyright (c) 2026 AlanBacker. Five modules (`audio`, `common`, `dem`, `geo`,
`tracks`) are vendored unmodified in `src/skyhum/_vendor/afg/`; the full licence text is in
`src/skyhum/_vendor/afg/LICENSE`. skyhum's locator (`src/skyhum/match/r5b.py`) builds on its method, which
implements colsto's approach.

**Runtime dependencies** (installed by pip, not vendored): NumPy (BSD-3-Clause), SciPy (BSD-3-Clause), soundfile
(BSD-3-Clause; bundles libsndfile, LGPL-2.1), webrtcvad-wheels (MIT; contains WebRTC VAD code, BSD-3-Clause),
optional PyTorch (BSD-3-Clause) and Matplotlib (PSF-based Matplotlib licence).

## Data

skyhum ships no recordings and no flight data. Results you produce with these sources carry their terms:

- **ADS-B history from adsb.lol** (https://adsb.lol): Open Database License (ODbL) 1.0. Attribute adsb.lol; a
  database you derive from it (track files, pass lists) must be shared under ODbL.
- **Australian Acoustic Observatory (A2O)** recordings (https://acousticobservatory.org), referenced by the
  benchmark: CC BY 4.0. Attribute the Australian Acoustic Observatory when you use or publish them.
- **xeno-canto** and **Freesound** recordings, referenced by link only: each recording keeps its own Creative
  Commons licence (several are NonCommercial or NoDerivatives).
- **Copernicus GLO-30 DEM** (only if you supply terrain tiles with `--dem-dir`): produced using Copernicus WorldDEM-30
  (c) DLR e.V. 2010-2014 and (c) Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European
  Union and ESA; licence terms at the Copernicus DEM distribution site.
