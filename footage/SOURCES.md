# Footage sources

All clips are from Wikimedia Commons, license read from the file page's license template (not the category).
Common processing for every clip:

- `<name>_src_seg.mp4`: the segment below, cut from the original at native resolution and fps, audio dropped, re-encoded H.264 (libx264 crf 14-16, yuv420p).
- `<name>_832x480_30fps.mp4`: from `_src_seg`, `fps=30` (frame drop/dup, no interpolation), scale to height 480 (lanczos, 853x480), center-crop to 832x480 (about 10 px trimmed per side; no stretch), H.264 crf 16.
- `<name>_frame0..3.jpg`: 4 evenly spaced frames of the 832x480 file.
- License-plate blur (where noted): plate box tracked with OpenCV CSRT from a hand-placed box, Gaussian blur over the padded box on each tracked frame, applied to `_src_seg` before the 832x480 derive.
- Originals kept in `_raw/` for provenance.

## car_rally_240sx (main car demo)

| | |
|---|---|
| File page | https://commons.wikimedia.org/wiki/File:240SX_Rally_car_at_the_2026_Southern_Ohio_Forest_Rally.webm |
| Author | HAV0Xunderscore (own work, Sony a6500) |
| License | CC0, "Creative Commons Zero, Public Domain Dedication", template `{{self|cc-zero}}` |
| Original | 1920x1080, 120 fps, 18.02 s, VP9 |
| Segment used | 7.0 s to 13.0 s (6.0 s, continuous, no cuts) |
| Processing | common steps above; 120 to 30 fps by frame drop; rear license plate blurred from about 1.7 s to the end of the segment (readable at 1080p otherwise) |
| Notes | Silver Nissan 240SX rally car comes round a gravel corner, slides past side-on with a dust plume, then pulls away. Panning handheld tracking. Visible: out-of-focus spectator caps at the bottom edge in the first ~0.5 s and a blurry hat at bottom center later (backs of heads, no faces); crew names and sponsor decals on the car. |

## car_jeep_dunes (second car option)

| | |
|---|---|
| File page | https://commons.wikimedia.org/wiki/File:Jeep_driving_on_Christmas_Valley_Sand_Dunes_(40688287560).webm |
| Author | BLM Oregon & Washington (video: Larisa Bogardus, BLM); via Flickr https://www.flickr.com/photos/blmoregon/40688287560 |
| License | Public domain, template `{{PD-USGov-BLM}}` (US federal government work, Bureau of Land Management) |
| Original | 1920x1080, 29.97 fps, 31.03 s, VP9 |
| Segment used | 9.0 s to 17.0 s (8.0 s, continuous) |
| Processing | common steps above; rear plate area blurred on all frames (partly readable at 1080p near the end) |
| Notes | Follow-cam chase shot of a lifted white Jeep climbing a sand dune under a blue sky. Smooth, clear motion, no people. |

## nature_waves_rocks_sanjuan (nature example)

| | |
|---|---|
| File page | https://commons.wikimedia.org/wiki/File:Crashing_waves_in_the_San_Juan_Islands_(40007700135).webm |
| Author | BLM Oregon & Washington (video: Nick Teague, BLM); via Flickr https://www.flickr.com/photos/blmoregon/40007700135 |
| License | Public domain, template `{{PD-USGov-BLM}}` |
| Original | 1920x1080, 29.97 fps, 65.23 s, VP9 |
| Segment used | 6.0 s to 14.0 s (8.0 s, continuous) |
| Processing | common steps above; no blur needed |
| Notes | Locked-off GoPro at Cattle Point: storm surge pours through a rock channel past driftwood logs, overcast sky. Water drops start landing on the lens after about 30 s in the original, so the segment stops well before that. |

## Rejected

| File | Why |
|---|---|
| Volkswagen T2.webm | CC0 tag on Commons, but sourced from Pixabay (excluded source). Its files `car_vw_t2_causeway_*` in this folder should be ignored/deleted. |
| Test Driving the Toyota Mirai.webm (PD-USGov-DOE) | many cuts, faces in close-up, title cards |
| Sights and Sounds- Fuji Speedway (PD-USGov-Military-Air Force) | graphics overlays, almost no moving cars |
| 2025-10-05 - Newport Antique Auto Hill Climb (PD-self) | crowds, tiny car, handheld zooms |
| Lohner Porsche at Gaisbergrennen 2011 (CC0) | people in focus, crowds |
| Indy5002025start (PD-self) | phone shot over a crowd |
| 2024 Indianapolis 500 - Sarah Stierch 05 (CC0) | crowd concourse, no cars |
| 2025-02-05 Retromobile Abarth 208 Boano 005 (CC0) | static indoor show car, people |
| Berlin Bundesautobahn 100 mov01 (CC0) | generic traffic from a bridge |
| USVI Sequence 01 2 (PD-USGov) | handheld walk-through, ends on a statue |
| Delaneys Falls (CC0) | thin, barely moving waterfall |
| Ocean surface waves (CC0) | low-drama open water |
