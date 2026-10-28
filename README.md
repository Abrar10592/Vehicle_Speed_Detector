# Vehicle Speed Detector

Vehicle speed detection for CCTV cameras (live streams) and recorded videos, with evidence clips, a live operator view and a calibration dashboard.

## Features
- Process recorded video files.
- Detect moving vehicles inside lane ROI.
- YOLO vehicle detection support.
- Track vehicles across frames.
- Estimate speed from calibrated measurement lines.
- Save one short evidence clip per speeding vehicle (only that vehicle is marked).
- Edit calibration data with a Streamlit dashboard.

## Current Camera Assumption
- One overhead camera centered above a 2-lane road.
- Each lane has:
	- a polygon ROI
	- a start line
	- an end line
	- known real distance between the two lines (meters)

## Project Structure
- [streamlit_app.py](streamlit_app.py): the app (landing page, Setup and Operator sections).
- [run_video.py](run_video.py): command-line processing (also what the app's monitoring runs).
- [src/pipeline.py](src/pipeline.py): main speed detection pipeline.
- [src/detector.py](src/detector.py): motion + YOLO vehicle detector.
- [src/tracker.py](src/tracker.py): centroid tracker.
- [src/evidence.py](src/evidence.py): evidence clip/metadata writer.
- [app/](app/): app pages: home, calibration, live monitor, evidence.
- [src/video_source.py](src/video_source.py): video file / live stream reading.
- [src/live.py](src/live.py), [src/monitor.py](src/monitor.py): live view files, monitoring service control, evidence index.
- [config/calibration.example.json](config/calibration.example.json): sample calibration config.

## Install
1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## The App
Start it:

```bash
streamlit run streamlit_app.py
```

The landing page leads to two sections:
- **Setup > Calibration**: cameras, their videos/streams, lanes, lines and speed limits (below).
- **Operator > Live monitor**: *Start monitoring* processes every camera in the background and shows each one live
  with speeds on screen; *Stop monitoring* finishes the clips being recorded, then stops.
- **Operator > Evidence**: clips of speeding vehicles by date and camera, with a player, downloads,
  *Open folder* (Windows Explorer) and the speed log.

### Calibration
- Select, add, remove and rename cameras in the sidebar; each camera has its own settings, video and lanes.
- Pick each camera's video with **Browse...** (its first frame is the drawing background).
- Select detector type (`motion` or `yolo`) and YOLO parameters.
- Define lane geometry for each lane:
	- polygon points
	- start line
	- end line
	- measured distance (meters)
- Preview overlays on a sample image/video frame.
- Save calibration JSON (example: `config/calibration.json`).

### Calibration tips
Where the lines are drawn decides how many vehicles get a speed:
- Put both lines where vehicles are big enough to be detected reliably (near/middle of the image). A vehicle only gets a speed if it is tracked across the start line and then the end line, so a line drawn in the far distance is crossed before the detector sees the vehicle.
- Draw each line across the full width of the carriageway and perpendicular to the traffic (in this camera view that is close to horizontal), so every lane travels the same distance between them.
- Draw the lane polygon over the whole carriageway, down to the bottom of the frame; anything outside it is masked out before detection.
- Measure `distance_m` along the road. Lane dashes are a convenient ruler: on UK motorways a dash plus its gap is 9 m, so lines placed on the ends of dashes N cycles apart are N x 9 m apart.

The default `config/calibration.json` is a calibration for `videos/sample_two_lane.mp4` built this way. The dashboard loads it on start; saving overwrites it.

## Live Cameras
Put a camera's stream address in its video field in Calibration instead of a file, e.g.
`rtsp://user:password@192.168.1.64:554/stream1`. The path after the port depends on the camera brand
(Hikvision: `/Streaming/Channels/101`, Dahua: `/cam/realmonitor?channel=1&subtype=0`); check the camera's manual.
A stream that drops out is reconnected automatically.

Processing speed matters for live streams. Without a GPU this computer processes about 9 frames per second per
camera with two cameras running. A live stream then has frames skipped (the newest frame is always processed, and
speeds are timed by capture time), which makes speeds less precise and fast vehicles harder to track. For live use,
run on a PC with an NVIDIA GPU (CUDA build of PyTorch) or use fewer cameras per PC. Recorded videos are not affected:
every frame is processed, the monitor just plays slower than real time.

## Run Recorded Video Processing

```bash
python run_video.py --video path/to/video.mp4 --config config/calibration.json --overlay-out output_overlay.mp4
```

Arguments:
- `--video` (optional): recorded input video path (default: the camera's video set in the dashboard).
- `--config`: calibration JSON path.
- `--camera` (optional): camera name in the config (default: the first camera), or `all` to process every camera at the same time.
  With `all`, each camera uses its own video and `--overlay-out output_overlay.mp4` writes `output_overlay_CAM01.mp4`, `output_overlay_CAM02.mp4`, ...;
  Ctrl+C stops all cameras and saves the clips recorded so far.
- `--overlay-out` (optional): output video with lane/line overlays and track info.
- `--debug-overlay` (optional): show per-vehicle lane membership and start/end crossing flags.
- `--live-dir` (optional): publish live frames and status there for the Live monitor (a `STOP` file there stops the run).
- `--realtime` (optional): process video files no faster than they were recorded.
- `--max-frames` (optional): stop after this many frames, for quick tests on long videos (e.g. `--max-frames 500` = first 20 s at 25 fps).

## Output
- Evidence clips are saved under the configured evidence folder:
	- `evidence/YYYY-MM-DD/CAMERA_ID/HH-MM-SS_<lane>_<speed>kmh.mp4` (one short clip per speeding vehicle)
- Speed event log CSV (one row per measured vehicle, with its camera):
	- `evidence/speed_events.csv`
	- a CSV from an older version with different columns is renamed to `speed_events_old.csv`

## Speed Formula
- $speed\_{km/h} = \frac{distance\_m}{time\_s} \times 3.6$

## Notes
- This version is optimized for low-traffic recorded videos.
- YOLO is recommended for better accuracy in dense traffic and low-light scenes.
- For production accuracy, perform lane-wise calibration after any camera movement.