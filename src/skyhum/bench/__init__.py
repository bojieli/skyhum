"""Benchmark sources: links and metadata only. Audio is never stored in the repository."""

SOURCES = [
    dict(id="xeno-canto", what="long field recordings with place and time", url="https://xeno-canto.org",
         licence="per recording; mostly CC BY-NC-SA, some CC BY-NC-ND (no derivatives), a few CC BY / CC0",
         note="items under ND licences are listed by link only; no labels derived from them are shipped"),
    dict(id="freesound", what="ambience recordings with place and time", url="https://freesound.org",
         licence="per sound: CC0, CC BY or CC BY-NC", note="API key needed for downloads; you supply your own"),
    dict(id="a2o", what="Australian Acoustic Observatory sensor recordings", url="https://acousticobservatory.org",
         licence="CC BY 4.0", note="rural sensors; the repeated-session results come from here"),
    dict(id="webcams", what="public airport and city webcam livestreams", url="https://www.youtube.com",
         licence="platform standard licence; not redistributable", note="video id and time only"),
    dict(id="events", what="public events and talks with known venue and time", url="https://www.youtube.com",
         licence="platform standard licence; not redistributable", note="video id and time only"),
    dict(id="adsb.lol", what="ADS-B flight history (the ground truth for every pass)", url="https://adsb.lol",
         licence="ODbL 1.0", note="attribute adsb.lol; derived databases must stay under ODbL"),
]
