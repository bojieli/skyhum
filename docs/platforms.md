# For meeting and streaming platforms

This page is for people who build call, meeting or streaming products. It lists what we tested and what reduced
the clue. "Measured" means real calls through the product; "simulated" means offline processing of recorded audio
with a reference implementation, not the product's own pipeline.

## Real calls (measured)

Known aircraft fly-overs from outdoor research sensors were played into real calls as the microphone signal, with
and without a person reading aloud, and the far end was recorded.

| Platform | No added voice | Reading aloud 60% of the time |
|---|---|---|
| Telegram Desktop | 17 of 19 flybys still recognised | 0 of 19 |
| Google Meet | 13 of 19 | 0 of 19 |
| Zoom (web client) | not measured: the test harness could not capture call audio from the web client | not measured |

A self-hosted LiveKit server was also tested as a plain WebRTC baseline; its numbers are not summarised here.
Mobile apps were not tested.

What this means: noise suppression and codecs in these products leave most of the low-frequency hum in place when
nobody is talking. Continuous speech masks it.

## What reduced the clue

| Measure | Where | Status | Effect (53 quiet outdoor hours, located within 2 km) |
|---|---|---|---|
| Nothing | - | simulated | 32 |
| Speech-gated low-band comfort noise below 400 Hz | sender side, in the platform | simulated | 4 |
| Doppler-tone tracking scrubber | before publishing a recording | simulated | 17 |
| Re-encoding alone | upload transcode | simulated | 32 |

The gate replaces the band below 400 Hz with comfort noise while the voice activity detector says nobody is
speaking, and passes everything untouched while someone speaks. It costs one FFT per 16 ms and about 100 ms of
latency in a streaming version. Neither scrubber cost measurable speech quality on two speech-quality tests. Both
reference implementations ship in this package (`skyhum scrub --method gate|tones`) so you can test them on your own
audio.

What none of these removes: a loud, close aircraft. It is broadband, partly opens the speech detector, and its
timing alone carries information.

## Contact

We have not contacted any platform about these results yet. If you work on one of the products above and want the
test details, see [SECURITY.md](../SECURITY.md) for how to reach the maintainers.
