"""Click parking-space polygons on the project's video and save them as JSON."""

import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).parent
VIDEO_PATH = PROJECT_DIR / "parking top view.mp4"
CONFIG_PATH = PROJECT_DIR / "data" / "parking_config.json"
WINDOW_NAME = "SmartVision Parking - Configure Spaces"


def load_existing_spaces():
    """Load a previous configuration so more spaces can be added to it."""
    if not CONFIG_PATH.exists():
        return []

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        return config.get("spaces", [])
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not read {CONFIG_PATH}: {error}")
        print("Starting with an empty configuration. Press r to clear, or q to quit.")
        return []


def save_spaces(spaces):
    """Write the configured parking spaces to the requested JSON file."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w", encoding="utf-8") as config_file:
        json.dump({"spaces": spaces}, config_file, indent=2)
    print(f"Saved {len(spaces)} parking spaces to: {CONFIG_PATH}")


def draw_polygon(frame, points, color, thickness=2):
    """Draw a closed polygon using OpenCV's pixel-coordinate format."""
    polygon = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [polygon], isClosed=True, color=color, thickness=thickness)


def draw_scene(frame, spaces, current_points, message):
    """Draw saved spaces, the in-progress polygon, and short instructions."""
    display = frame.copy()

    for space in spaces:
        points = space["polygon"]
        draw_polygon(display, points, (0, 255, 255))
        label_x, label_y = points[0]
        cv2.putText(
            display,
            space["id"],
            (label_x, max(18, label_y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if current_points:
        for point in current_points:
            cv2.circle(display, tuple(point), 4, (255, 0, 255), -1)
        if len(current_points) >= 2:
            open_line = np.array(current_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [open_line], False, (255, 0, 255), 2)

    # A dark strip keeps instructions readable over both cars and road.
    cv2.rectangle(display, (0, 0), (frame.shape[1], 43), (35, 35, 35), -1)
    cv2.putText(
        display,
        "Click corners | Enter: finish | u: undo | c: cancel | r: reset | s: save | q: quit",
        (8, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(display, message, (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
    return display


def main():
    video = cv2.VideoCapture(str(VIDEO_PATH))
    if not video.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    has_frame, frame = video.read()
    video.release()
    if not has_frame:
        print(f"Could not read a frame from: {VIDEO_PATH}")
        return

    spaces = load_existing_spaces()
    current_points = []
    message = f"Next space: P{len(spaces) + 1:02d} | Click at least 3 corners, then press Enter."
    mouse_state = {"points": current_points}

    def on_mouse(event, x, y, flags, parameter):
        # The frame is shown at its original size, so mouse x/y are video pixels.
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_state["points"].append([int(x), int(y)])

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    print("Click the corners of a parking space in order around its edge.")
    print("Press Enter to finish that space. Press s to save, or q to quit without saving.")

    while True:
        current_points = mouse_state["points"]
        display = draw_scene(frame, spaces, current_points, message)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (10, 13):  # Enter: close the current polygon and assign its ID.
            if len(current_points) < 3:
                message = "A polygon needs at least 3 clicked corners."
            else:
                space_id = f"P{len(spaces) + 1:02d}"
                spaces.append({"id": space_id, "zone": "A", "polygon": current_points.copy()})
                mouse_state["points"] = []
                message = f"{space_id} added | Next space: P{len(spaces) + 1:02d}"
                print(f"Added {space_id} with {len(spaces[-1]['polygon'])} points.")
        elif key == ord("u"):
            if current_points:
                current_points.pop()
                message = "Removed the last corner from the current polygon."
            elif spaces:
                removed = spaces.pop()
                message = f"Removed {removed['id']}. You can click it again."
            else:
                message = "There is no point or parking space to undo."
        elif key == ord("c"):
            mouse_state["points"] = []
            message = f"Current polygon cancelled | Next space: P{len(spaces) + 1:02d}"
        elif key == ord("r"):
            spaces.clear()
            mouse_state["points"] = []
            message = "All spaces cleared in memory. Press s to save the empty configuration."
        elif key == ord("s"):
            if current_points:
                message = "Finish or cancel the current polygon before saving."
            elif not spaces:
                message = "No spaces to save. Click at least 3 corners and press Enter first."
            else:
                save_spaces(spaces)
                message = f"Saved {len(spaces)} spaces. You may continue editing or press q to quit."
        elif key == ord("q") or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    print("Configuration window closed.")


if __name__ == "__main__":
    main()
