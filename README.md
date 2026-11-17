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
- [server.py](server.py), [update.py](update.py), [selftest.py](selftest.py), Start.bat, Update.bat: unattended running and updates (below).

## Install
Needs Python 3.14 and Git. In the project folder:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.lock.txt
```

[requirements.lock.txt](requirements.lock.txt) holds the exact package versions the project was tested with, so every
computer gets the same ones. [requirements.txt](requirements.txt) lists only the direct dependencies with minimum
versions; after upgrading or adding a package, refresh the lock with `pip freeze > requirements.lock.txt`.

Videos and evidence clips are not stored in Git. On a new computer, copy the `testvideo` and `videos` folders into the
project folder by hand. Start the app from the project folder: video paths inside it are saved in
`config/calibration.json` relative to it (e.g. `testvideo/50-1.MTS`).

### NVIDIA graphics card (optional)
The install above gives the processor-only PyTorch. On a computer with an NVIDIA card, swap in the CUDA build of the
same version: the detector then uses the card by itself and runs about twice as fast (see Live Cameras). Stop the app
first; the download is 2.6 GB.

```powershell
pip install "torch==2.14.1+cu126" "torchvision==0.29.1+cu126" --index-url https://download.pytorch.org/whl/cu126
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The second command must print `True` and the card's name (tested: GeForce GTX 1650, NVIDIA driver 555.97; an older
driver may need updating). `pip freeze` on that computer lists `torch==2.14.1+cu126`, which the normal package index
does not have: when refreshing the lock there, keep its `torch` and `torchvision` lines without `+cu126`.

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
- Pick each camera's video with **Browse...** (mp4, avi, mov, mkv, or camcorder mts/m2ts; its first frame is the drawing background).
- Select detector type (`motion` or `yolo`) and YOLO parameters.
- Define lane geometry for each lane in the full-width **Draw** section:
	- lane area: click each corner, then click the first corner again to close the polygon (an unclosed polygon is not saved)
	- start line (blue) and end line (red): press at one end and drag to the other
	- measured distance (meters)
- **Place the line ends exactly** (below the drawing, once a line is saved): choose a line end and click in the magnified picture exactly where the line must touch the road. Do this for every end of both lines; see the first calibration tip.
- **Drawing size** scales the frame to fit the screen (800 px fits a laptop); coordinates are always saved at the video's full resolution. The toolbar above the drawing has Undo, Redo, Edit (move or resize a shape) and Reset.
- Save calibration JSON (example: `config/calibration.json`).

### Calibration tips
Where the lines are drawn decides how accurate the speeds are and how many vehicles get one:
- Mark both ends of each line on the road (a cone or paint mark on each side) and put each line end exactly where its mark meets the ground, with **Place the line ends exactly**. Far from a low camera one pixel covers a long stretch of road: in a trial with a camera about 2 m high and a line 28 m away, a line drawn 20 pixels too high (through the middle of the cone instead of its base) added about 4 m to a 20 m distance, and every speed read about 15% low.
- Vehicles are timed at the point where they touch the road (the bottom center of their box), so draw lanes and lines on the road surface.
- Put both lines where vehicles are big enough to be detected reliably (near/middle of the image), and where the whole vehicle, wheels included, is still inside the picture. A vehicle only gets a speed if it is tracked across both lines, so a line drawn in the far distance is crossed before the detector sees the vehicle, and a line at the edge of the picture is crossed after the vehicle's box has been cut off.
- The lines work in both directions: whichever one a vehicle crosses first starts the timer. Each lane has one speed limit, so give opposite directions separate lanes if their limits differ.
- Prefer a straight stretch of road: a vehicle that bends between the lines travels a different distance.
- Calibration belongs to one camera position. If the camera is moved, zoomed or knocked, even slightly, redo it.
- Draw each line across the full width of the carriageway and perpendicular to the traffic (in this camera view that is close to horizontal), so every lane travels the same distance between them.
- Draw the lane polygon over the whole carriageway, down to the bottom of the frame; vehicles whose wheels are outside it are ignored.
- Measure `distance_m` along the road. Lane dashes are a convenient ruler: on UK motorways a dash plus its gap is 9 m, so lines placed on the ends of dashes N cycles apart are N x 9 m apart.

`config/calibration.json` holds this computer's cameras, including their passwords, and is not stored in Git. The dashboard loads it on start and saving overwrites it; on a new computer it is created by the first save. [config/calibration.example.json](config/calibration.example.json) is a sample, a calibration for `videos/sample_two_lane.mp4` built this way.

## Running on a Server
On a computer that runs unattended, start the app with **Start.bat** instead of `streamlit run`. It runs
[server.py](server.py), which:
- starts the app (http://localhost:8501) and starts it again if it ends;
- starts monitoring again after a crash, a power cut or an update, if it was on (*Start monitoring* pressed and
  *Stop monitoring* not pressed);
- restarts the app after an update.

To start it whenever the Windows user logs in, run once:

```powershell
.venv\Scripts\python.exe server.py --install-autostart
```

(`--remove-autostart` undoes it.) To come back by itself after a restart of the computer, Windows must also log
that user in automatically.

## Updates
A server only installs **released versions**: Git tags named like `v1.2`. Work pushed to `main` without a tag is
not installed anywhere. To release the current `main`:

```powershell
git tag v1.2
git push origin v1.2
```

On the server, **Setup > Update** shows the installed version; *Check for updates* looks for a newer release and
*Update now* installs it. An update:
1. stops monitoring (clips being recorded are saved first) and keeps a copy of `config/calibration.json`;
2. installs the release, and the Python packages if `requirements.lock.txt` changed;
3. runs [selftest.py](selftest.py): it measures a generated video of a box moving at a known speed, loads this
   computer's calibration and runs the vehicle detector;
4. if anything failed, goes back to the version that was running;
5. starts monitoring again, and the app is restarted (by Start.bat) to use the new version.

If the app does not open, double-click **Update.bat**: it does the same without the app, and also reinstalls the
Python packages and tests the installed version when there is no newer release. `python update.py --undo` goes
back to the version before the last update. Both write to `runtime/update.log`.

## Live Cameras
In Calibration, press **Live camera...** next to the video field and enter the camera's name, IP address, user name
and password. **Connect** checks that the camera answers, then fills in its stream address and shows its picture for
drawing the lanes (**Refresh camera picture** fetches a new one). Hikvision and Dahua are in the brand list; for
another brand choose *Other* and enter the stream path from the camera's manual.

The stream address can also be typed into the video field by hand, e.g.
`rtsp://user:password@192.168.1.64:554/stream1` (Hikvision path: `/Streaming/Channels/101`, Dahua:
`/cam/realmonitor?channel=1&subtype=0`). A stream that drops out is reconnected automatically.

Processing speed matters for live streams. When a camera sends more frames than the computer processes, frames are
skipped (the newest frame is always processed, and speeds are timed by capture time), which makes speeds less precise
and fast vehicles harder to track. Each camera's *fps* in Live monitor should reach the camera's own frame rate.
Recorded videos are not affected: every frame is processed, the monitor just plays slower than real time.

A line is crossed between two processed frames; the moment is worked out from the vehicle's position before and
after, so a lower frame rate costs little accuracy (a recorded pass read 34.4 km/h at 25 fps and 34.2 km/h at 12.5).

Small, distant vehicles: the detector searches the area around the lanes instead of the whole frame, and when one
line is much shorter in the picture than the other (much farther away), the surroundings of that far line are
searched a second time at full resolution. Without it a motorcycle 28 m from a 4-megapixel camera was only found
after it had passed the far line and got no speed. The second search takes time: about a third fewer frames per
second on such a camera. A far line that is closer to the camera, or a camera with a longer lens, needs it less.

The table below was measured before the second search was added, on the development laptop (Core i5-9300H,
GeForce GTX 1650) with 1080p 25 fps video, in frames per second per camera:

| Cameras running together | Processor only | With the NVIDIA card |
|---|---|---|
| 1 | 14.6 | 30.6 |
| 2 | 9.6 | 23.5 |
| 4 | 5.3 | 12.1 |

With the card, the processor is the limit (video decoding, drawing, evidence clips): the card stays under half busy
and uses under 1 GB of its memory. For more cameras at full frame rate, use a PC with more processor cores or fewer
cameras per PC.

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
- Speeds depend on the frame rate read from the video. For camcorder `.mts` files, compare the `frame rate` line printed at the start of a run with the file's frame rate in MediaInfo; if they differ, set the correct value with **Use FPS override** in Calibration.
- YOLO is recommended for better accuracy in dense traffic and low-light scenes.
- For production accuracy, perform lane-wise calibration after any camera movement.