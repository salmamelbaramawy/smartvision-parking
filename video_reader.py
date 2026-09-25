"""Play the project video one frame at a time using OpenCV."""

from pathlib import Path

import cv2


VIDEO_PATH = Path(__file__).parent / "parking top view.mp4"


def main():
    # VideoCapture opens the video file so OpenCV can read its frames in order.
    video = cv2.VideoCapture(str(VIDEO_PATH))

    if not video.isOpened():
        print(f"Could not open video: {VIDEO_PATH}")
        return

    print("Playing video. Press q in the video window to quit.")

    while True:
        # read() returns whether a frame was read and the frame image itself.
        has_frame, frame = video.read()

        if not has_frame:
            # No more frames means the video has reached its end.
            break

        cv2.imshow("SmartVision Parking - Video Reader", frame)

        # Wait briefly so the window can update; q closes playback.
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()
    print("Video playback finished.")


if __name__ == "__main__":
    main()
