"""Vendored subset of afg (AlanBacker/audio-flyover-geolocation, MIT, commit 4c2465b): audio, dem, geo, tracks,
common. Unmodified copies; see LICENSE in this directory. dem.fetch_copernicus downloads terrain tiles and is never
called by skyhum unless you pass --dem-dir pointing at a directory without GeoTIFFs (refused by the CLI)."""
