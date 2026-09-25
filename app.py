# """Streamlit interface for the existing edge-density parking detector."""

# import cv2
# import streamlit as st

# from edge_density_detector import (
#     PROJECT_DIR,
#     THRESHOLD,
#     create_polygon_mask,
#     load_parking_spaces,
#     process_frame,
# )


# st.set_page_config(page_title="SmartVision Parking", page_icon="🅿️", layout="wide")

# st.title("🅿️ SmartVision Parking")
# st.subheader("AI-Based Smart Parking Occupancy Detection System")
# st.write(
#     "Watch the configured parking spaces as the video is analyzed. "
#     "Green spaces are FREE; red spaces are OCCUPIED."
# )

# spaces = load_parking_spaces()
# video_path = PROJECT_DIR / "parking top view.mp4"

# with st.sidebar:
#     st.header("About this analysis")
#     st.write("**Method:** OpenCV Canny edge detection and edge density")
#     st.write(f"**Threshold:** {THRESHOLD:.2f}")
#     st.write(f"**Configured spaces:** {len(spaces)}")
#     st.write(f"**Input video:** {video_path.name}")
#     st.divider()
#     st.subheader("Independent validation sample")
#     st.write("Reviewed on 31 unseen frames and 14 spaces (434 observations).")
#     st.write("**Accuracy:** 97.0%")
#     st.write("**Precision:** 95.2%")
#     st.write("**Recall:** 99.6%")
#     st.caption(
#         "These results describe the validation sample, not a guarantee for every "
#         "frame or every parking space. Shadows, markings, and surface texture can affect edge density."
#     )
#     st.divider()
#     st.markdown("🟩 **FREE** &nbsp;&nbsp; 🟥 **OCCUPIED**", unsafe_allow_html=True)

# if "completed" not in st.session_state:
#     st.session_state.completed = False
# if "last_counts" not in st.session_state:
#     st.session_state.last_counts = None
# if "last_results" not in st.session_state:
#     st.session_state.last_results = None

# button_col1, button_col2 = st.columns([1, 1])
# with button_col1:
#     start_clicked = st.button(
#         "▶️ Start Detection",
#         type="primary",
#         disabled=st.session_state.completed,
#         width="stretch",
#     )
# with button_col2:
#     reset_clicked = st.button("↻ Reset", width="stretch")

# if reset_clicked:
#     st.session_state.completed = False
#     st.session_state.last_counts = None
#     st.session_state.last_results = None
#     st.rerun()

# metric_cols = st.columns(4)
# total_metric = metric_cols[0].empty()
# occupied_metric = metric_cols[1].empty()
# free_metric = metric_cols[2].empty()
# rate_metric = metric_cols[3].empty()

# st.markdown("### Live parking video")
# frame_placeholder = st.empty()
# progress_placeholder = st.empty()
# st.markdown("### Parking-space status")
# status_placeholder = st.empty()


# def show_counts(counts):
#     total, occupied, free = counts
#     total_metric.metric("Total Spaces", total)
#     occupied_metric.metric("Occupied", occupied)
#     free_metric.metric("Free", free)
#     rate_metric.metric("Occupancy Rate", f"{occupied / total * 100:.1f}%" if total else "0.0%")


# def show_status_table(results):
#     rows = [
#         {
#             "Space ID": space["id"],
#             "Status": "🟩 FREE" if results[space["id"]]["status"] == "FREE" else "🟥 OCCUPIED",
#             "Edge Density": f"{results[space['id']]['edge_density']:.3f}",
#         }
#         for space in spaces
#     ]
#     status_placeholder.table(rows)


# if start_clicked:
#     if not video_path.is_file():
#         st.error(f"Parking video not found: {video_path}")
#     else:
#         capture = cv2.VideoCapture(str(video_path))
#         if not capture.isOpened():
#             st.error(f"Could not open the parking video: {video_path}")
#         else:
#             total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
#             progress_bar = progress_placeholder.progress(0.0, text="Preparing video…")
#             frame_number = 0
#             masks = None
#             try:
#                 while True:
#                     has_frame, frame = capture.read()
#                     if not has_frame:
#                         break

#                     if masks is None:
#                         masks = {
#                             space["id"]: create_polygon_mask(space, frame.shape)
#                             for space in spaces
#                         }

#                     # The shared detector performs grayscale conversion, Canny,
#                     # per-polygon measurement, classification, and drawing once.
#                     display_frame, results, counts = process_frame(frame, spaces, masks)
#                     frame_number += 1
#                     st.session_state.last_counts = counts
#                     st.session_state.last_results = results

#                     frame_placeholder.image(
#                         display_frame,
#                         channels="BGR",
#                         width="stretch",
#                         caption=f"Frame {frame_number} of {total_frames}",
#                     )
#                     show_counts(counts)
#                     show_status_table(results)
#                     if total_frames:
#                         progress_bar.progress(
#                             min(frame_number / total_frames, 1.0),
#                             text=f"Analyzing frame {frame_number} of {total_frames}",
#                         )
#             finally:
#                 capture.release()

#             if frame_number == 0:
#                 st.error("The video opened but did not contain a readable frame.")
#             else:
#                 st.session_state.completed = True
#                 progress_bar.progress(1.0, text=f"Complete — analyzed {frame_number} frames")
#                 st.success("Video analysis complete. Select Reset to run it again.")
# elif st.session_state.last_counts and st.session_state.last_results:
#     show_counts(st.session_state.last_counts)
#     show_status_table(st.session_state.last_results)
# else:
#     total_metric.metric("Total Spaces", len(spaces))
#     occupied_metric.metric("Occupied", "—")
#     free_metric.metric("Free", "—")
#     rate_metric.metric("Occupancy Rate", "—")
#     st.info("Select **Start Detection** to analyze the parking video.")

# if not start_clicked and not st.session_state.last_results:
#     st.caption("Space IDs, current status, and edge density will appear here during analysis.")       


"""Streamlit interface for the existing edge-density parking detector."""

import cv2
import time
import streamlit as st

from edge_density_detector import (
    PROJECT_DIR,
    THRESHOLD,
    create_polygon_mask,
    load_parking_spaces,
    process_frame,
)


st.set_page_config(
    page_title="SmartVision Parking",
    page_icon="🅿️",
    layout="wide",
)

st.title("🅿️ SmartVision Parking")
st.subheader("AI-Based Smart Parking Occupancy Detection System")

st.write(
    "Watch the configured parking spaces as the video is analyzed. "
    "Green spaces are FREE; red spaces are OCCUPIED."
)

spaces = load_parking_spaces()
video_path = PROJECT_DIR / "parking top view.mp4"


# =========================
# Sidebar
# =========================

with st.sidebar:
    st.header("About this analysis")

    st.write("**Method:** OpenCV Canny edge detection and edge density")
    st.write(f"**Threshold:** {THRESHOLD:.2f}")
    st.write(f"**Configured spaces:** {len(spaces)}")
    st.write(f"**Input video:** {video_path.name}")

    st.divider()

    st.subheader("Independent validation sample")

    st.write("Reviewed on 31 unseen frames and 14 spaces (434 observations).")
    st.write("**Accuracy:** 97.0%")
    st.write("**Precision:** 95.2%")
    st.write("**Recall:** 99.6%")

    st.caption(
        "These results describe the validation sample, not a guarantee for every "
        "frame or every parking space. Shadows, markings, and surface texture can "
        "affect edge density."
    )

    st.divider()

    st.markdown(
        "🟩 **FREE** &nbsp;&nbsp; 🟥 **OCCUPIED**",
        unsafe_allow_html=True,
    )


# =========================
# Session state
# =========================

if "completed" not in st.session_state:
    st.session_state.completed = False

if "last_counts" not in st.session_state:
    st.session_state.last_counts = None

if "last_results" not in st.session_state:
    st.session_state.last_results = None


# =========================
# Buttons
# =========================

button_col1, button_col2 = st.columns([1, 1])

with button_col1:
    start_clicked = st.button(
        "▶️ Start Detection",
        type="primary",
        disabled=st.session_state.completed,
        width="stretch",
    )

with button_col2:
    reset_clicked = st.button(
        "↻ Reset",
        width="stretch",
    )


# =========================
# Reset
# =========================

if reset_clicked:
    st.session_state.completed = False
    st.session_state.last_counts = None
    st.session_state.last_results = None
    st.rerun()


# =========================
# UI placeholders
# =========================

metric_cols = st.columns(4)

total_metric = metric_cols[0].empty()
occupied_metric = metric_cols[1].empty()
free_metric = metric_cols[2].empty()
rate_metric = metric_cols[3].empty()


st.markdown("### Live parking video")

frame_placeholder = st.empty()
progress_placeholder = st.empty()

st.markdown("### Parking-space status")

status_placeholder = st.empty()


# =========================
# Helper functions
# =========================

def show_counts(counts):
    total, occupied, free = counts

    total_metric.metric(
        "Total Spaces",
        total,
    )

    occupied_metric.metric(
        "Occupied",
        occupied,
    )

    free_metric.metric(
        "Free",
        free,
    )

    rate_metric.metric(
        "Occupancy Rate",
        f"{occupied / total * 100:.1f}%" if total else "0.0%",
    )


def show_status_table(results):
    rows = [
        {
            "Space ID": space["id"],
            "Status": (
                "🟩 FREE"
                if results[space["id"]]["status"] == "FREE"
                else "🟥 OCCUPIED"
            ),
            "Edge Density": (
                f"{results[space['id']]['edge_density']:.3f}"
            ),
        }
        for space in spaces
    ]

    status_placeholder.table(rows)


# =========================
# Start Detection
# =========================

if start_clicked:

    if not video_path.is_file():

        st.error(
            f"Parking video not found: {video_path}"
        )

    else:

        capture = cv2.VideoCapture(
            str(video_path)
        )

        if not capture.isOpened():

            st.error(
                f"Could not open the parking video: {video_path}"
            )

        else:

            # Get original video FPS
            fps = capture.get(
                cv2.CAP_PROP_FPS
            )

            # Safety fallback
            if not fps or fps <= 0:
                fps = 30.0

            # Time between frames
            frame_delay = 1.0 / fps

            total_frames = int(
                capture.get(
                    cv2.CAP_PROP_FRAME_COUNT
                )
            )

            progress_bar = progress_placeholder.progress(
                0.0,
                text="Preparing video…",
            )

            frame_number = 0
            masks = None

            try:

                while True:

                    # Start timing this frame
                    frame_start_time = time.time()

                    has_frame, frame = capture.read()

                    if not has_frame:
                        break

                    # Create masks only once
                    if masks is None:

                        masks = {
                            space["id"]: create_polygon_mask(
                                space,
                                frame.shape,
                            )
                            for space in spaces
                        }

                    # Process current frame
                    display_frame, results, counts = process_frame(
                        frame,
                        spaces,
                        masks,
                    )

                    frame_number += 1

                    st.session_state.last_counts = counts
                    st.session_state.last_results = results

                    # Display frame
                    frame_placeholder.image(
                        display_frame,
                        channels="BGR",
                        width="stretch",
                        caption=(
                            f"Frame {frame_number} "
                            f"of {total_frames}"
                        ),
                    )

                    # Update counters
                    show_counts(counts)

                    # Update parking status
                    show_status_table(results)

                    # Update progress
                    if total_frames:

                        progress_bar.progress(
                            min(
                                frame_number / total_frames,
                                1.0,
                            ),
                            text=(
                                f"Analyzing frame "
                                f"{frame_number} of "
                                f"{total_frames}"
                            ),
                        )

                    # ---------------------------------
                    # Keep approximately original FPS
                    # ---------------------------------

                    elapsed = (
                        time.time()
                        - frame_start_time
                    )

                    remaining = (
                        frame_delay
                        - elapsed
                    )

                    if remaining > 0:
                        time.sleep(remaining)

            finally:

                capture.release()

            # =========================
            # Finished
            # =========================

            if frame_number == 0:

                st.error(
                    "The video opened but did not "
                    "contain a readable frame."
                )

            else:

                st.session_state.completed = True

                progress_bar.progress(
                    1.0,
                    text=(
                        f"Complete — analyzed "
                        f"{frame_number} frames"
                    ),
                )

                st.success(
                    "Video analysis complete. "
                    "Select Reset to run it again."
                )


# =========================
# Show previous results
# =========================

elif (
    st.session_state.last_counts
    and st.session_state.last_results
):

    show_counts(
        st.session_state.last_counts
    )

    show_status_table(
        st.session_state.last_results
    )


# =========================
# Initial state
# =========================

else:

    total_metric.metric(
        "Total Spaces",
        len(spaces),
    )

    occupied_metric.metric(
        "Occupied",
        "—",
    )

    free_metric.metric(
        "Free",
        "—",
    )

    rate_metric.metric(
        "Occupancy Rate",
        "—",
    )

    st.info(
        "Select **Start Detection** "
        "to analyze the parking video."
    )


if (
    not start_clicked
    and not st.session_state.last_results
):

    st.caption(
        "Space IDs, current status, and edge density "
        "will appear here during analysis."
    )
