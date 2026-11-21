"""Loud-event channel: loud, close aircraft passes as timing evidence (research channel, inference only).

  features   blind event detector on 8 or 16 kHz audio: third-octave levels, floor tracking, envelope peaks,
             Lorentzian width, band spectrum, modulation and a high-frequency de-Doppler fit (PyTorch, CPU)
  model      prep_event (speech / ATC gating), featurize (96 features), the none / jet / prop / rotor classifier
             (numpy export, no pickle) and the calibrated per-event evidence (models/loud/calibration.json)
  locate     known-clock localisation from events and ADS-B tracks: single-event mirror points, multi-event grid
             score with a shared clock (PyTorch, CPU)

Not wired into the command line in this build. The channel's results and limits are in the research write-up; the
classifier's training data and licence are in models/MODEL_CARD.md.
"""
