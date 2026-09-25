"""Detect cars and match their centers to configured parking spaces."""

import json
import os
from pathlib import Path

import cv2
import numpy as np


PROJECT_DIR = Path(__file__).parent
VIDEO_PATH = PROJECT_DIR / "parking top view.mp4"
MODEL_PATH = PROJECT_DIR / "yolo11n.pt"
CONFIG_PATH = PROJECT_DIR / "data" / "parking_config.json"
CAR_CLASS_ID = 2  # In the model's COCO classes, 2 means car.
CONFIDENCE_THRESHOLD = 0.10
IMAGE_SIZE = 1280

# Keep Ultralytics settings inside the project instead of the user profile.
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_DIR))

from ultralytics import YOLO


def load_parking_spaces():
    """Read the manually configured parking-space polygons from JSON."""
    with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)
    return config["spaces"]


def draw_cars_and_get_centers(result, model, frame, rotated_clockwise=False):
    """Draw each car box and return its center in the original frame."""
    car_centers = []

    if result.boxes is None:
        return car_centers

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        if rotated_clockwise:
            # Convert the box from the rotated detection frame back to the video.
            rotated_x1, rotated_y1 = x1, y1
            x1 = rotated_y1
            y1 = frame.shape[0] - 1 - x2
            x2 = y2
            y2 = frame.shape[0] - 1 - rotated_x1
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        car_centers.append((center_x, center_y))

        top_left = (int(x1), int(y1))
        bottom_right = (int(x2), int(y2))
        center_pixel = (int(center_x), int(center_y))
        # Cyan car boxes and yellow center dots are distinct from space colors.
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 0), 2)
        cv2.circle(frame, center_pixel, 3, (0, 255, 255), -1)
        label = f"{class_name} {confidence:.2f}"
        label_y = max(20, int(y1) - 8)
        cv2.putText(
            frame,
            label,
            (int(x1), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return car_centers


def match_cars_to_spaces(spaces, car_centers):
    """Mark a space occupied when any car center falls inside its polygon."""
    space_statuses = {}

    for space in spaces:
        polygon = np.array(space["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        is_occupied = False

        for center in car_centers:
            # Returns 1 inside, 0 on the edge, and -1 outside the polygon.
            if cv2.pointPolygonTest(polygon, center, False) >= 0:
                is_occupied = True
                break

        space_statuses[space["id"]] = is_occupied

    return space_statuses


def draw_spaces_and_counts(frame, spaces, space_statuses):
    """Draw green FREE and red OCCUPIED polygons, IDs, and frame counts."""
    occupied_count = sum(space_statuses.values())
    total_count = len(spaces)
    free_count = total_count - occupied_count

    for space in spaces:
        points = np.array(space["polygon"], dtype=np.int32).reshape((-1, 1, 2))
        is_occupied = space_statuses[space["id"]]
        color = (0, 0, 255) if is_occupied else (0, 200, 0)
        cv2.polylines(frame, [points], True, color, 2)

        # The mean of the clicked corners gives a readable label position.
        label_x = int(np.mean(points[:, 0, 0]))
        label_y = int(np.mean(points[:, 0, 1]))
        label_position = (label_x, label_y)
        cv2.putText(frame, space["id"], label_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, space["id"], label_position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Put the count panel on the road area so it stays separate from the spaces.
    overlay = frame.copy()
    cv2.rectangle(overlay, (6, 98), (230, 177), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "Parking occupancy", (14, 119), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Total spaces: {total_count}", (14, 139),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Occupied: {occupied_count}", (14, 157),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Free: {free_count}", (14, 174),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 0), 1, cv2.LINE_AA)

    return total_count, occupied_count, free_count


def main():
    # Ultralytics downloads the pretrained weights the first time they are needed.
    model = YOLO(MODEL_PATH)
    spaces = load_parking_spaces()

    video = cv2.VideoCapture(str(VIDEO_PATH))
    if not video.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    print("Detecting cars and matching parking spaces. Press q to quit.")
    window_name = "SmartVision Parking - Occupancy Matching"
    last_counts = (len(spaces), 0, len(spaces))

    while True:
        has_frame, frame = video.read()
        if not has_frame:
            break

        # Rotating clockwise makes more of the parking-row cars detectable.
        detection_frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Keep the best tested CPU settings for these small top-view cars.
        # classes=[2] tells YOLO to return only the COCO car class.
        predictions = model.predict(
            source=detection_frame,
            classes=[CAR_CLASS_ID],
            conf=CONFIDENCE_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device="cpu",
            augment=True,
            verbose=False,
        )

        result = predictions[0]
        car_centers = draw_cars_and_get_centers(
            result, model, frame, rotated_clockwise=True
        )
        statuses = match_cars_to_spaces(spaces, car_centers)
        last_counts = draw_spaces_and_counts(frame, spaces, statuses)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()
    total_count, occupied_count, free_count = last_counts
    print("Video finished.")
    print(f"Final frame counts: Total={total_count}, Occupied={occupied_count}, Free={free_count}")


if __name__ == "__main__":
    main()
