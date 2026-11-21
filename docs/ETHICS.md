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
- **Statement, not a gate.** skyhum is for research, validation and defence. It is not for locating private
  individuals. The code does not try to enforce this: anyone can edit open-source code, and a gate would only
  stand in the way of people checking the published numbers.
- **Honest output.** `find` and `locate` print a point estimate only above the strict threshold, and every result
  states the base rate and that the confidence has not been validated.
- **Reproducibility.** The whole-country search, the metro clustering and the evaluators are published so that every
  number in the write-up can be checked (REPRODUCING.md). The country search is an explicit command
  (`skyhum locate --country`), never the default: verify and audit stay the front door.
- **Local only.** Audio never leaves the machine. There is no telemetry.
- **No recordings in the repository.** The benchmark lists links and metadata and respects each recording's
  licence.
- **Honest output.** Every result says which matcher produced it, what was not tested, and that `UNCORROBORATED`
  is not proof of anything.

## What actually limits misuse

The low base rate: one blind hour names the right US region for about 5% of public recordings. The risk is
repetition from one spot and a listener who already has a guess, which is exactly what people should test on their
own audio.

## Reporting misuse or a safety problem

See [SECURITY.md](../SECURITY.md). Reports about the gate design, harmful use cases we missed, or wording that
overstates what the tool can do are all welcome.
