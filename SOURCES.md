# Sources for the Video editing / bridge PR media

## car_rally_240sx (Phase 1 bridge probe, Unsloth Studio end-to-end run, UI render)

| | |
|---|---|
| File page | https://commons.wikimedia.org/wiki/File:240SX_Rally_car_at_the_2026_Southern_Ohio_Forest_Rally.webm |
| Author | HAV0Xunderscore (own work) |
| License | CC0 (`{{self|cc-zero}}`) |
| Segment | 7.0 s to 13.0 s of the original, rear plate blurred (details in `../simtx/footage/SOURCES.md`) |
| Here | `work/rally_24fps_832x480.mp4`: that segment at fps=24 (frame drop), 832x480, 144 frames. Original in `_raw/`. |

## NodeCore "Wood Tech" (SIM -> REAL strip)

| | |
|---|---|
| File page | https://commons.wikimedia.org/wiki/File:NodeCore-_Wood_Tech.webm |
| Author | Warr1024 (NodeCore, a Minetest game); uploaded to Commons from the author's YouTube video |
| License | CC BY 3.0, so the restyled strip and anything containing it carries the credit "NodeCore: Wood Tech, by Warr1024, CC BY 3.0, via Wikimedia Commons" |
| Segment | 228.5 s to 236.5 s (8 s of steady walking, no menus) |
| Processing | crop 728x420 at (276,100) to drop the HUD, scale to 832x480, 30 fps, H.264; `nc_832.mp4` on the VM |
| Output | `demo/sim2real*.mp4` is a derivative (Cosmos3-Nano-Sim-Transfer restyle); credit as above |

## Drone chase (demo input)

Generated, not footage: Cosmos3-Nano text-to-video through Unsloth Studio, prompt in `scripts/demo_api.py` (`DRONE`),
seeds 0 to 3, 121 frames at 832x480 / 24 fps. Seed 2 used (smoothest, worst step 20.7 dB). `demo/t2v/`.

## Long-gap bridge showcase (longgap/)

Generated, not footage. The 34 s original is the seed-2 drone clip extended by six Cosmos3-Nano continuations (seeds 100 to 105, 121 frames each) through Unsloth Studio; the gap fills are new continuations and a bridge from different seeds. `scripts/longgap.py`.

## Jeep long-gap showcase (longgap/jeep/)

| | |
|---|---|
| File page | https://commons.wikimedia.org/wiki/File:Jeep_driving_on_Christmas_Valley_Sand_Dunes_(40688287560).webm |
| Author | BLM Oregon & Washington (video: Larisa Bogardus, BLM); via Flickr https://www.flickr.com/photos/blmoregon/40688287560 |
| License | Public domain, `{{PD-USGov-BLM}}` (US federal government work, Bureau of Land Management) |
| Used | the whole clip, 0 to 31.0 s; one continuous shot (ffmpeg scene detection at 0.25 found no cuts) |
| Processing | rear plate blurred on every frame (CSRT track of the Jeep's rear from a hand-placed box, Gaussian blur over the padded plate area; the tracker never lost it), then fps=24 by frame drop, scale to height 480, centre-crop to 832x480, H.264 |

## Rejected

| What | Why |
|---|---|
| Commons "Aerial drone chase of rally cars" 1 and 2, "Aerial views of a car driving in snow" | CC BY 3.0 re-uploads of YouTube videos, outside the CC0 / PD / US-gov rule for footage |
| Minetest 5.6.1 trailer clip (CC BY-SA 3.0) | backup only; NodeCore worked, so not used |
