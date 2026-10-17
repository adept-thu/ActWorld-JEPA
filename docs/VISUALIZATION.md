# NAVSIM video: what is shown

The video uses **R383's saved NAVSIM v1 submission predictions** and recorded
OpenScene/nuPlan camera frames. It is an offline, open-loop visualization, not
a real-world deployment or a closed-loop simulation. No new model training or
inference was performed to make it.

- Three clips, 21 consecutive frames each; 63 source frames / **31.5 seconds**.
- Source cadence and playback are **2 Hz, real-time speed**. Frames are not
  generated or temporally interpolated; an H.264 container may repeat frames
  for browser compatibility, without introducing new observations.
- Orange/red-to-yellow points show the saved **4-second predicted trajectory**
  (eight future steps). They are not a ground-truth trajectory.
- The front view is recorded imagery. The BEV uses the current map and annotated
  actor boxes; those boxes are **not model perception output**.
- The trajectory uses the native rear-axle ego coordinate system. No arbitrary
  rotation, scale adjustment or smoothing is used to improve visual alignment.
- Clips were selected for turns and visible traffic with complete consecutive
  predictions, not as a representative performance estimate.
- Likely plate/face regions are pixelated using projected current actor boxes.
  This is a visualization treatment, not a model input or an evaluation change.

The source prediction file SHA256 is
`ff482967953a5de886b03638c953c24fb7740d3f62ebc15867b515f6fd69998d`.
See [video provenance](assets/navsim-video-provenance.json).

## Render your own video

Use the NAVSIM v1 environment from the [v1 guide](NAVSIM_V1_R383.md).
Generate `submission.pkl` first, or use your own trusted NAVSIM v1 submission.
Create a JSON file with entries of this form:

```json
[{"log": "your_log.pkl", "start": 10, "end": 30}]
```

`start` and `end` are inclusive indices in that log. Every frame must have a
saved prediction and a front image, and timestamps must be consecutive at 2 Hz.
The renderer stops on missing predictions instead of inventing a trajectory.

From the repository root:

```bash
python tools/render_navsim_video.py \
  --submission /absolute/path/to/submission.pkl \
  --logs "$OPENSCENE_DATA_ROOT/navsim_logs/test" \
  --sensors "$OPENSCENE_DATA_ROOT/sensor_blobs/test" \
  --maps "$NUPLAN_MAPS_ROOT" \
  --clips /absolute/path/to/clips.json \
  --output /absolute/path/to/new-video-directory
```

The renderer uses CPU only, saves an MPEG-4 preview, sample frames and provenance.
For broad browser playback, convert it with FFmpeg:

```bash
ffmpeg -i navsim_replay.mp4 -c:v libx264 -crf 20 -pix_fmt yuv420p \
  -movflags +faststart -an navsim-r383-demo.mp4
```

## Media attribution and terms

The derived NAVSIM video and poster use nuPlan/Motional data redistributed by
OpenScene. They retain **CC BY-NC-SA 4.0** and the **nuPlan Dataset License
Agreement for Non-Commercial Use**, as stated by the
[OpenScene dataset provider](https://github.com/OpenDriveLab/OpenScene#license-and-citation).
Attribution: Motional/nuPlan and OpenScene contributors; trajectory overlays,
selection, masking and layout by the ActWorld-JEPA authors.

These media are **not covered by the repository's Apache-2.0 code license**.
Do not infer commercial-use permission from the code license. Consult the
[dataset terms](https://www.nuscenes.org/terms-of-use) and
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Raw logs, sensor archives and benchmark predictions are not redistributed here.
