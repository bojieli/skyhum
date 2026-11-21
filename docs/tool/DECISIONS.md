# Decisions log

Decisions taken while preparing the public repository, with the reason. Owner decisions are marked as such.

| Date | Decision | Why |
|---|---|---|
| 2026-11-21 | Publish a fresh squashed repository; keep the research history private | The research history held infrastructure details and data that could identify private individuals |
| 2026-11-21 | Licence MIT (owner) | Matches the existing LICENSE and the upstream afg code |
| 2026-11-21 | Remove the `find` gate, attestation, area cap and local log (owner) | The repository must let anyone check every published number; a statement of intended use replaces the gate |
| 2026-11-21 | Ship detector v3, the round-5b tables and the loud classifier with a full source and licence list (owner) | The published numbers should reproduce with the original models; the maintainers treat training as non-derivative use |
| 2026-11-21 | Loud classifier ships as numpy tree arrays, not a pickle | A pickle can run code when loaded; the export is exact on 400 dev events |
| 2026-11-21 | Whole-country search ships as `skyhum locate --country`, not the default (owner) | `verify` and `audit` stay the front door; every result prints the base rate |
| 2026-11-21 | Webcam audio is released only as derived features (owner); event and speech livestreams are never released | Livestreams cannot be re-fetched; events and speech could identify private individuals |
| 2026-11-22 | Feature set 3 (30-400 Hz magnitude spectrograms, 2 s loudness curves) approved for release (owner); wider sets wait for an intelligibility test and the owner's go | Speech-band features need evidence, not reasoning, that speech cannot be recovered |
| 2026-11-22 | Saved-outputs bundle includes the 6 xeno-canto box items | Public recordings, not event, speech or private streams; needed for exact reproduction of A17d/A17f/clk30m |
