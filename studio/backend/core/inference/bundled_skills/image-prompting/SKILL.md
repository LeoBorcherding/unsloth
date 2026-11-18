---
name: image-prompting
description: Write detailed prompts for an image or video model when the user asks to generate one.
allowed-tools: generate_image generate_video use_library_model
---

# Write an image or video prompt

Use this skill when the user asks for an image or a video and their request is short or vague.

1. Read the generate_image or generate_video tool description: when it carries a tip for the model loaded in Unsloth Studio and that tip disagrees with a step below, follow the tip.
2. Name the subject first, with the details that must stay the same across shots.
3. Add the setting: place, time of day and weather.
4. Pick a style: photo, illustration, painting or 3D render, plus the lens or medium when it matters.
5. Describe the light and the mood.
6. Set the composition: framing and camera angle, and for a video how the camera and the subject move.
7. Put any words that must appear in the picture in quotes.
8. For more shots of the same character, repeat the subject's details word for word.

## Picking a model that is not loaded

use_library_model loads a downloaded image or video model for one image or clip, then puts the chat
model back. Each call reloads both models, so it is slow.

- Use generate_image or generate_video when offered and the user did not name another model.
- Use use_library_model when the user names an image or video model, or none is loaded. For a video
  model it also takes seconds.
- Call it without a model first if you do not know the downloaded ids, then pick one by name.
- Ask for several images in one prompt only if the model supports it; otherwise each image is a full swap.
- unload_chat_model picks what happens to the chat model: true unloads it for the image and loads it
  back, false keeps it loaded (and refuses a model that does not fit beside it), auto keeps it only
  when both fit. Leave it out to follow the user's Keep multiple models loaded setting.

Keep the user's own words and intent. Add no people, brands or text they did not ask for, and leave steps and guidance out unless the user asks, so the model's own defaults apply.
