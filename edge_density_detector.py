"""Classify configured parking spaces with OpenCV edge density."""

import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_DIR / "parking top view.mp4"
CONFIG_PATH = PROJECT_DIR / "data" / "parking_config.json"
THRESHOLD = 0.16

# These are the same Canny settings and one-pixel mask erosion used in validation.
CANNY_LOW = 60
CANNY_HIGH = 140
MASK_EROSION_SIZE = 3


def load_parking_spaces():
    """Load the manually configured parking polygons from the existing JSON."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config["spaces"]


def create_polygon_mask(space, frame_shape):
    """Create a boolean mask containing only interior pixels of one polygon."""
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    polygon = np.asarray(space["polygon"], dtype=np.int32)
    cv2.fillPoly(mask, [polygon], 255)

    # Exclude the polygon border so parking-line edges count less often.
    kernel = np.ones((MASK_EROSION_SIZE, MASK_EROSION_SIZE), dtype=np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    return mask > 0


def calculate_edge_density(edge_map, polygon_mask):
    """Return edge-pixel count divided by valid pixels inside the polygon mask."""
    inside_pixels = edge_map[polygon_mask]
    total_pixels = inside_pixels.size
    if total_pixels == 0:
        raise ValueError("A parking polygon has no interior pixels in the video frame.")

    edge_pixels = int(np.count_nonzero(inside_pixels))
    return edge_pixels / total_pixels


def classify_space(edge_density):
    """Return OCCUPIED above the validated threshold; otherwise return FREE."""
    return "OCCUPIED" if edge_density > THRESHOLD else "FREE"


def process_frame(frame, spaces, polygon_masks):
    """Measure all spaces, draw their results, and return counts and scores."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edge_map = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    results = {}
    occupied_count = 0

    for space in spaces:
        space_id = space["id"]
        density = calculate_edge_density(edge_map, polygon_masks[space_id])
        status = classify_space(density)
        results[space_id] = {"status": status, "edge_density": density}
        occupied_count += status == "OCCUPIED"

        points = np.asarray(space["polygon"], dtype=np.int32)
        color = (0, 0, 255) if status == "OCCUPIED" else (0, 200, 0)
        cv2.polylines(frame, [points], isClosed=True, color=color, thickness=2)

        # Put only the short ID on the video so nearby labels do not overlap.
        center_x = int(np.mean(points[:, 0]))
        center_y = int(np.mean(points[:, 1]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        text_width = cv2.getTextSize(space_id, font, font_scale, 1)[0][0]
        x = max(0, center_x - text_width // 2)
        y = max(10, center_y + 4)
        cv2.putText(frame, space_id, (x, y), font, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, space_id, (x, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    total_count = len(spaces)
    free_count = total_count - occupied_count
    if len(results) != total_count or occupied_count + free_count != total_count:
        raise RuntimeError("Space evaluation or occupancy count invariant failed.")

    # Keep the count panel in the open roadway below the top row of spaces.
    panel = frame.copy()
    cv2.rectangle(panel, (7, 102), (228, 211), (0, 0, 0), thickness=-1)
    cv2.addWeighted(panel, 0.72, frame, 0.28, 0, frame)
    cv2.putText(frame, "Edge-density occupancy", (15, 122), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Total: {total_count}", (15, 145), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Occupied: {occupied_count}", (15, 169), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Free: {free_count}", (15, 193), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Threshold: {THRESHOLD:.2f}", (15, 207), cv2.FONT_HERSHEY_SIMPLEX,
                0.36, (255, 255, 255), 1, cv2.LINE_AA)

    return frame, results, (total_count, occupied_count, free_count)


def create_status_panel(results, counts):
    """Build a separate readable list of each space's status and edge score."""
    total_count, occupied_count, free_count = counts
    panel = np.full((382, 510, 3), (38, 38, 38), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(panel, "Parking space status", (10, 21), font, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, f"Total {total_count}   Occupied {occupied_count}   Free {free_count}",
                (10, 43), font, 0.37, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(panel, "ID      STATUS      EDGE DENSITY", (10, 64), font, 0.34,
                (190, 190, 190), 1, cv2.LINE_AA)

    # Two columns keep all 41 results readable in a panel beside the video.
    items = list(results.items())
    split_at = (len(items) + 1) // 2
    columns = (items[:split_at], items[split_at:])
    for column_index, column_items in enumerate(columns):
        x = 10 + column_index * 252
        for row_index, (space_id, result) in enumerate(column_items):
            y = 82 + row_index * 14
            status = result["status"]
            status_color = (0, 200, 0) if status == "FREE" else (60, 60, 255)
            cv2.putText(panel, space_id, (x, y), font, 0.35,
                        (245, 245, 245), 1, cv2.LINE_AA)
            cv2.putText(panel, status, (x + 42, y), font, 0.35,
                        status_color, 1, cv2.LINE_AA)
            cv2.putText(panel, f"{result['edge_density']:.3f}", (x + 126, y), font, 0.35,
                        (220, 220, 220), 1, cv2.LINE_AA)
    return panel


def main():
    spaces = load_parking_spaces()
    video = cv2.VideoCapture(str(VIDEO_PATH))
    if not video.isOpened():
        raise FileNotFoundError(f"Could not open parking video: {VIDEO_PATH}")

    first_ok, first_frame = video.read()
    if not first_ok:
        video.release()
        raise RuntimeError("The parking video opened but did not contain a readable frame.")
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Masks are created once and reused for each frame; each contains polygon pixels only.
    polygon_masks = {
        space["id"]: create_polygon_mask(space, first_frame.shape)
        for space in spaces
    }
    video.set(cv2.CAP_PROP_POS_FRAMES, 0)

    frame_index = 0
    total_space_evaluations = 0
    last_counts = (len(spaces), 0, len(spaces))
    window_name = "Parking Occupancy - Edge Density"
    status_window_name = "Parking Space Status and Edge Scores"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(status_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, first_frame.shape[1], first_frame.shape[0])
    cv2.resizeWindow(status_window_name, 510, 382)
    cv2.moveWindow(window_name, 0, 35)
    cv2.moveWindow(status_window_name, first_frame.shape[1], 35)

    print(f"Video: {VIDEO_PATH.name}")
    print(f"Parking spaces: {len(spaces)}; threshold: {THRESHOLD:.2f}")
    print("Playing the full video. Press q in the video window to quit early.")

    try:
        while True:
            has_frame, frame = video.read()
            if not has_frame:
                break

            display_frame, results, last_counts = process_frame(
                frame, spaces, polygon_masks
            )
            status_panel = create_status_panel(results, last_counts)
            total_space_evaluations += len(results)

            if (frame_index + 1) % 100 == 0:
                total, occupied, free = last_counts
                print(
                    f"Processed {frame_index + 1} frames; "
                    f"latest counts: Total={total}, Occupied={occupied}, Free={free}"
                )

            cv2.imshow(window_name, display_frame)
            cv2.imshow(status_window_name, status_panel)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                print("Playback stopped by user before the end of the video.")
                break
            frame_index += 1
    finally:
        video.release()
        cv2.destroyAllWindows()

    total, occupied, free = last_counts
    print(f"Frames processed: {frame_index}")
    print(f"Space evaluations: {total_space_evaluations}")
    print(f"Final frame counts: Total={total}, Occupied={occupied}, Free={free}")
    print(f"Count invariant: {occupied} + {free} = {occupied + free} (Total {total})")
    if frame_index and total_space_evaluations == frame_index * len(spaces):
        print("Verification passed: all configured spaces were evaluated on every frame.")
    else:
        print("Verification failed: not all spaces were evaluated on every processed frame.")


if __name__ == "__main__":
    main()
