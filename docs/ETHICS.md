# Ethics and responsible use

## What skyhum is for

- Checking what your own recordings, streams and calls reveal about where you were.
- Testing whether a fix (muting, the pause gate, tone scrubbing) removes the clue before you publish.
- Verifying a location claim attached to public media, as part of journalism or research.
- Research on public recordings or recordings whose subjects agreed to it.

## What it is not for

Do not use skyhum to locate private individuals, to track anyone, or to harass anyone. Do not run `find` on audio
from someone who has not agreed to it. If you are unsure whether a use is acceptable, it probably is not.

## Why ship a locator-grade checker at all

People who stream, call or publish recordings from a fixed place cannot judge their exposure with a tool that only
says "nothing found". To answer "would my audio give me away?" honestly, the checker has to do the same matching an
adversary would. The method is already public (see the credits in the README), so withholding a defensive tool
mostly withholds it from defenders.

The measured risk is modest for a single blind recording and larger for repeated sessions from one spot and for a
listener who already has a guess ([limits.md](limits.md)). The design follows from that.

## Design choices

- **Verify and audit are the front door.** They answer a question about a place you already name.
- **find is gated.** It needs `--research` and an attestation (`own`, `consented`, `public-benchmark`), it refuses
  areas more than 100 km across (refused, not truncated), and it prints a point estimate only above the strict
  threshold. Every attempt, allowed or refused, is logged locally with a hash of the audio.
- **No whole-country search.** The research code that searches a whole country, clusters metros, and scores many
  sessions across a country is not part of this package.
- **Local only.** Audio never leaves the machine. There is no telemetry.
- **No recordings in the repository.** The benchmark lists links and metadata and respects each recording's
  licence.
- **Honest output.** Every result says which matcher produced it, what was not tested, and that `UNCORROBORATED`
  is not proof of anything.

## These are speed bumps

Anyone can edit open-source code and remove a gate. The gates exist so the default behaviour matches the purpose
and so misuse takes deliberate effort. The real limits are the low blind success rate, the area cap and the absence
of a country search.

## Reporting misuse or a safety problem

See [SECURITY.md](../SECURITY.md). Reports about the gate design, harmful use cases we missed, or wording that
overstates what the tool can do are all welcome.
