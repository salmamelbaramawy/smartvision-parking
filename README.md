# SmartVision Parking

**AI-Based Smart Parking Occupancy Detection System**

SmartVision Parking analyzes a fixed, top-view parking video and classifies each manually configured parking space as **FREE** or **OCCUPIED**. The final occupancy detector uses OpenCV image processing; it does not use YOLO to make occupancy decisions.

## How it works

For each video frame, the detector:

1. Reads the frame with OpenCV.
2. Converts it to grayscale.
3. Applies Canny edge detection.
4. Counts edge pixels only inside each configured parking-space polygon.
5. Divides the edge count by the number of valid pixels in that polygon to get edge density.
6. Applies the fixed threshold: density **greater than 0.16** is **OCCUPIED**; otherwise it is **FREE**.
7. Draws the results and reports total, occupied, and free spaces.

The polygon masks use the interior pixels of each polygon, with a one-pixel border erosion to reduce the effect of painted parking lines. The system currently evaluates 41 configured spaces.

## Project files

```text
parking detection/
├── app.py                          # Main Streamlit application
├── edge_density_detector.py       # Final edge-density occupancy detector
├── parking_configurator.py        # Manual parking-polygon configuration tool
├── video_reader.py                # Simple video playback tool
├── detector.py                    # Earlier YOLO experiment; not used by final detector
├── requirements.txt               # Python dependencies, including Streamlit
├── .gitignore                      # Excludes local environments and experiment outputs
├── README.md
├── parking top view.mp4            # Input parking video
├── yolo11n.pt                      # Earlier YOLO experiment weights
├── Ultralytics/                    # Earlier YOLO settings
└── data/
    └── parking_config.json         # The 41 configured parking spaces
```

The JPEG previews and `edge_yolo_independent_validation.csv` in the project root are experiment outputs. The final detector does not read them, and `.gitignore` excludes them from uploads. `.venv/` is the local Python environment, and `__pycache__/` contains automatically generated Python cache files; those are excluded too.

## Requirements and installation

Python 3 and pip are required. From PowerShell, open the project folder and create the environment if it does not already exist:

```powershell
py -m venv .venv
```

Install the project dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`numpy` and `opencv-python` are used by the final detector and the configuration/video tools. `ultralytics` remains in the requirements because the preserved older `detector.py` experiment imports it; the final edge-density detector does not use it.

## Run the main application

From the project folder, start the Streamlit app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens in a browser. Select **Start Detection** to analyze the video. It shows the current frame with the configured polygons, live Total/Occupied/Free counts and occupancy rate, plus the status and edge density for all 41 spaces. Select **Reset** to clear the displayed run and start again.

The sidebar documents the OpenCV edge-density method, fixed 0.16 threshold, input video, space count, and held-out validation results. The displayed validation metrics describe the reviewed sample and are not a guarantee for every frame or space.

## Streamlit Community Cloud Deployment

1. Push this project to a GitHub repository.
2. Open Streamlit Community Cloud and connect that GitHub repository.
3. Select the repository's `main` branch.
4. Set the app's main file path to `app.py`.
5. Select **Deploy**.

`app.py` is the main Streamlit application. `edge_density_detector.py` is the standalone Computer Vision detector and also provides the shared frame-processing functions used by the app. The occupancy method remains OpenCV Canny edge detection plus edge density, using threshold **0.16**. Keep `parking top view.mp4` and `data/parking_config.json` in the repository; both are runtime inputs.

## Run the standalone detector

The standalone OpenCV detector remains available independently of the Streamlit app:

Run this from the project folder:

```powershell
.\.venv\Scripts\python.exe edge_density_detector.py
```

The program opens the parking video and a separate status panel. The video shows green FREE polygons, red OCCUPIED polygons, and each parking-space ID. The status panel lists the status and edge-density score for all spaces, along with the counts. Press **q** in a video window to quit early. This script is the same processing engine used by `app.py`.

## Parking-space configuration

The manually configured spaces are stored in `data/parking_config.json`. It currently contains 41 polygons with IDs P01–P41. The detector reads this file without changing it. `parking_configurator.py` is the separate tool for manually drawing or editing polygons; save changes there only when you intend to change the parking configuration.

## Validation

On an independent held-out sample of 31 unseen video frames and 14 manually reviewed spaces (434 space-frame observations), the edge-density method at threshold 0.16 achieved:

- Accuracy: **97.0%**
- Precision: **95.2%**
- Recall: **99.6%**

These figures summarize the sampled observations. They are not a claim that every frame or all 41 spaces will have those accuracy values.

## Limitations

Edge density can be affected by shadows, painted or floor markings, and surface texture inside a parking polygon. A car that is only partly visible while entering or leaving can also be classified incorrectly. The threshold is a single fixed value for all spaces.
