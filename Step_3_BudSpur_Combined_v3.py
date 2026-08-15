# coding: utf-8

# ============================================================
# STEP 3: FIRST-YEAR-WOOD WINTER-BUD COUNTING
# MALE + FIRST-YEAR CUTIE MASK TRACKING / REGIONAL LK BUD TRACKING
# ============================================================
#
# COUNTING RULE
#   - Male cane and countable first-year wood are segmented independently.
#   - Each stream has its own Cutie InferenceCore temporal memory.
#   - Male cane has priority wherever the two cane masks overlap.
#   - Countable cane = first-year Cutie mask minus male Cutie mask.
#   - New winter buds may start only on the countable cane mask.
#   - Existing confirmed buds may reconnect through a brief mask miss.
#   - New buds are confirmed in 2 of 3 frames and then counted permanently.
#   - Spurs are NOT detected or tracked in this version.
#
# ORIENTATION NORMALISATION
#   - Decoded 2160 x 3840 portrait frames are rotated 90 degrees clockwise.
#   - Decoded 3840 x 2160 landscape frames are left unchanged.
#   - The complete pipeline then uses 3840 x 2160 landscape coordinates.
#   - --include_roi and --include_roi_ranges therefore use landscape coordinates.
#
# BUD TRACKING
#   - YOLO bud detection.
#   - Same-frame duplicate suppression.
#   - Regional Lucas-Kanade optical flow only; no local per-bud LK flow.
#   - Temporary cane centrelines are rebuilt each frame as matching clues.
#   - Hungarian one-to-one matching.
#   - 8-frame lost-track memory.
#   - 2-of-3-frame confirmation.
#
# OUTPUT COMPATIBILITY
#   - The Step 3 CSV column structure is unchanged.
#   - All spur columns are retained and written as zero.
#   - bud_final_* columns mirror the immediate permanent bud counts.
#   - Legacy combined count columns mirror bud counts.
#   - processed_frames/frame_XXXXXX.jpg is retained for Step 6.
#   - Processed JPGs show the original frame, bud boxes only, active ROI lines,
#     and the frame-ID/count summary. No bud IDs, confidence text, cane masks,
#     or spurs are drawn on the processed JPGs.
#   - Optional diagnostic videos can be enabled from the run code:
#       Video 1 = male + first-year Cutie masks + bud boxes/IDs + summary.
#       Video 2 = original video + bud boxes/IDs + summary.
# ============================================================

import argparse
import os
import sys
import cv2
import time
import math
from collections import deque

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from ultralytics import YOLO

try:
    from scipy.optimize import linear_sum_assignment
except ImportError as exc:
    raise ImportError(
        "This pipeline requires SciPy for Hungarian bud matching. "
        "Install it with: pip install scipy"
    ) from exc


# ============================================================
# 1) RUNTIME ARGUMENTS AND PATHS
# ============================================================


def parse_bool(value):
    """Parse a command-line true/false value."""
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"1", "true", "yes", "y", "on"}:
        return True

    if text in {"0", "false", "no", "n", "off"}:
        return False

    raise argparse.ArgumentTypeError(
        f"Expected a boolean value such as true/false. Received: {value!r}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run Step 3 winter-bud counting with male/first-year Cutie cane "
            "tracking and regional optical-flow bud tracking."
        )
    )

    parser.add_argument("--video_path", default="Barry Roderick-vid2.mp4")
    parser.add_argument("--cane_model_path", default="CaneY26V10.pt")
    parser.add_argument("--bud_model_path", default="WinBudy12n.pt")
    parser.add_argument("--male_model_path", default="Male_Yolo_26_Final.pt")
    parser.add_argument(
        "--cutie_repo",
        required=True,
        help="Path to the Cutie repository/install root.",
    )

    parser.add_argument("--output_dir", default="CaneMotionTracker_Test")
    parser.add_argument(
        "--output_video",
        default=None,
        help=(
            "Optional path for diagnostic video 1 when --save_output_videos "
            "is true. Video 1 shows male and first-year Cutie masks plus "
            "winter-bud boxes/IDs and the frame summary. When omitted, a "
            "default path is created inside --output_dir."
        ),
    )
    parser.add_argument(
        "--output_bud_only_video",
        default=None,
        help=(
            "Optional path for diagnostic video 2 when --save_output_videos "
            "is true. Video 2 shows the original unhighlighted frame plus "
            "winter-bud boxes/IDs and the frame summary. When omitted, the "
            "path is derived from --output_video using a '_bud_only' suffix."
        ),
    )
    parser.add_argument(
        "--save_output_videos",
        type=parse_bool,
        default=False,
        help=(
            "Create the two optional diagnostic MP4 videos. "
            "Accepts true/false. Default: false."
        ),
    )
    parser.add_argument("--output_frame_csv", default=None)
    parser.add_argument("--processed_frames_dir", default=None)

    parser.add_argument(
        "--include_roi",
        default=None,
        help=(
            "Optional default/fallback counting ROI in normalised landscape "
            "3840x2160 coordinates using 'x1,y1,x2,y2'. Portrait 2160x3840 "
            "frames are rotated 90 degrees clockwise before this ROI is used. "
            "It applies to the full video when --include_roi_ranges is omitted. "
            "When both arguments are supplied, it is used only for frames not "
            "covered by a configured range."
        ),
    )
    parser.add_argument(
        "--include_roi_ranges",
        default=None,
        help=(
            "Optional frame-specific counting ROIs in normalised landscape "
            "3840x2160 coordinates. Use semicolon-separated entries in the form "
            "'start_frame-end_frame:x1,y1,x2,y2'. Example: "
            "'0-1200:500,200,3300,2000;1201-2500:300,150,3500,2050'. "
            "Portrait 2160x3840 frames are rotated 90 degrees clockwise before "
            "the ROI is used. When ranges are supplied without a fallback "
            "--include_roi, frames outside all ranges are not countable."
        ),
    )
    parser.add_argument(
        "--save_processed_frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save annotated JPG frames for Step 6 bay-preview clips.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional frame limit. By default the full video is processed.",
    )

    return parser.parse_args()


ARGS = parse_args()

VIDEO_PATH = ARGS.video_path
CANE_MODEL_PATH = ARGS.cane_model_path
BUD_MODEL_PATH = ARGS.bud_model_path
MALE_MODEL_PATH = ARGS.male_model_path
CUTIE_REPO = os.path.abspath(os.path.expanduser(ARGS.cutie_repo))

OUTPUT_DIR = ARGS.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

VIDEO_STEM = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
OUTPUT_FRAME_CSV = ARGS.output_frame_csv or os.path.join(
    OUTPUT_DIR,
    f"{VIDEO_STEM}_combined_frame_analysis.csv",
)
PROCESSED_FRAMES_DIR = ARGS.processed_frames_dir or os.path.join(
    OUTPUT_DIR,
    "processed_frames",
)

SAVE_OUTPUT_VIDEOS = bool(ARGS.save_output_videos)

OUTPUT_VIDEO = ARGS.output_video or os.path.join(
    OUTPUT_DIR,
    f"{VIDEO_STEM}_male_first_year_buds.mp4",
)

if ARGS.output_bud_only_video:
    OUTPUT_BUD_ONLY_VIDEO = ARGS.output_bud_only_video
else:
    output_video_root, output_video_ext = os.path.splitext(OUTPUT_VIDEO)
    if not output_video_ext:
        output_video_ext = ".mp4"
    OUTPUT_BUD_ONLY_VIDEO = (
        f"{output_video_root}_bud_only{output_video_ext}"
    )

SAVE_PROCESSED_FRAMES = ARGS.save_processed_frames
INCLUDE_ROI_TEXT = ARGS.include_roi
INCLUDE_ROI_RANGES_TEXT = ARGS.include_roi_ranges
MAX_FRAMES_TO_PROCESS = ARGS.max_frames

if SAVE_PROCESSED_FRAMES:
    os.makedirs(PROCESSED_FRAMES_DIR, exist_ok=True)

if not os.path.isdir(CUTIE_REPO):
    raise FileNotFoundError(f"Cutie repository was not found: {CUTIE_REPO}")

if CUTIE_REPO not in sys.path:
    sys.path.insert(0, CUTIE_REPO)

try:
    from cutie.inference.inference_core import InferenceCore
    from cutie.utils.get_default_model import get_default_model
except Exception as exc:
    raise ImportError(
        "Cutie could not be imported.\n"
        f"Configured repository: {CUTIE_REPO}\n"
        "Confirm that Cutie is installed and importable in the current "
        "SageMaker kernel/environment.\n"
        f"Original import error: {exc}"
    ) from exc


# ============================================================
# 2) MODEL AND TRACKING SETTINGS
# ============================================================

# Male-cane YOLO.
MALE_CONF_THRES = 0.30
MALE_IOU_THRES = 0.40
MALE_IMG_SIZE = 960
MALE_MAX_DET = 1000
MALE_CLASS_ID = 0

# First-year-cane YOLO.
CANE_CONF_THRES = 0.35
CANE_IOU_THRES = 0.70
CANE_IMG_SIZE = 1280
CANE_MAX_DET = 300
FIRST_YEAR_CLASS_NAME = "countable_1st_year_cane"
CANE_MIN_MASK_AREA = 250
CANE_YOLO_CLOSE_KERNEL = 5
CANE_YOLO_CLOSE_ITER = 1

# Cutie cane-mask tracking.
CUTIE_OBJECT_ID = 1
CUTIE_MAX_INTERNAL_SIZE = 720
CUTIE_CORRECTION_INTERVAL = 5
CUTIE_MIN_SEED_PIXELS = 250
CUTIE_USE_AMP = True
CUTIE_FUSION_CLOSE_KERNEL = 3
CUTIE_FUSION_CLOSE_ITER = 1
CUTIE_EMPTY_RESET_FRAMES = 10

# Bud YOLO and cane acceptance.
BUD_CLASS_ID = 0
BUD_CONF_THRES = 0.30
BUD_IOU_THRES = 0.80
BUD_IMG_SIZE = 1280
BUD_MAX_DET = 1000
MIN_BUD_BOX_MASK_OVERLAP = 0.08
USE_BUD_CENTER_INSIDE_MASK = True
BUD_BOX_EXPAND_FOR_MASK = 4
CANE_ACCEPT_DILATE_KERNEL = 5
CANE_ACCEPT_DILATE_ITER = 1
SAME_FRAME_DUP_STRONG_IOU = 0.50
SAME_FRAME_DUP_WEAK_IOU = 0.15
SAME_FRAME_DUP_CENTER_DIST = 12

# Temporary cane structures for bud matching.
CANE_COMPONENT_MIN_AREA = 250
CANE_COMPONENT_CLOSE_KERNEL = 7
CENTERLINE_BIN_SIZE = 18
CENTERLINE_MIN_POINTS_PER_BIN = 5
CENTERLINE_SMOOTH_WINDOW = 3
CANE_ASSIGN_MAX_DISTANCE = 70.0

# Regional optical flow only.
CANE_CORRIDOR_RADIUS = 30
CANE_CORRIDOR_RESOLUTION_SCALE = 0.50
REGION_GRID_COLS = 8
REGION_GRID_ROWS = 6
REGION_MAX_CORNERS = 4000
REGION_QUALITY_LEVEL = 0.005
REGION_MIN_DISTANCE = 5
REGION_BLOCK_SIZE = 5
REGION_MIN_POINTS_PER_CELL = 5
REGION_MAX_POINTS_PER_CELL = 250
REGION_FB_MAX_ERROR = 3.5
REGION_OUTLIER_MAD_MULT = 3.5
REGION_LK_WIN_SIZE = 51
REGION_LK_MAX_LEVEL = 4
REGION_FILL_NEIGHBOUR_RADIUS = 2
REGION_MAX_REASONABLE_DISPLACEMENT = 180.0
GLOBAL_AFFINE_RANSAC_THRESHOLD = 4.0
REGION_FLOW_RESOLUTION_SCALE = 0.50

# Confirmed-track prediction and Hungarian matching.
LOST_MAX_GAP_FRAMES = 8
PREDICTION_MIN_RADIUS = 24.0
PREDICTION_MAX_RADIUS = 150.0
PREDICTION_BOX_DIAG_MULT = 2.2
PREDICTION_REGION_WEAK_EXTRA = 35.0
PREDICTION_GAP_EXTRA_PER_FRAME = 14.0
MATCH_MAX_AREA_RATIO = 3.5
MATCH_MAX_COST = 1.15
MATCH_LARGE_COST = 1e6
MATCH_WEIGHT_DISTANCE = 0.52
MATCH_WEIGHT_AREA = 0.13
MATCH_WEIGHT_IOU = 0.10
MATCH_WEIGHT_CANE = 0.10
MATCH_WEIGHT_ORDER = 0.08
MATCH_WEIGHT_PATCH = 0.07
PATCH_EXPAND = 10
PATCH_SIZE = 32
ALLOW_CONFIRMED_RECONNECT_OFF_MASK = True

# New-bud 2-of-3 confirmation.
PENDING_CONFIRM_HITS = 2
PENDING_CONFIRM_WINDOW = 3
PENDING_MAX_AGE_FRAMES = 4
PENDING_MAX_MISSES = 2
PENDING_MATCH_RADIUS = 75.0
PENDING_MATCH_MAX_AREA_RATIO = 3.5
PENDING_MATCH_MAX_COST = 1.10

# Processed-frame and optional-video drawing.
BUD_BOX_THICKNESS = 2
BUD_NEW_COLOR = (0, 0, 255)       # red
BUD_OLD_COLOR = (0, 255, 0)       # green
BUD_WAITING_COLOR = (0, 165, 255) # orange

# Active counting ROI line shown on processed JPGs.
# Only the rectangle is drawn; no ROI label or grey outside-mask is added.
ROI_LINE_COLOR = (235, 235, 235)
ROI_LINE_THICKNESS = 3

# Optional diagnostic-video cane-mask appearance, matching the test pipeline.
MALE_COLOR = (120, 200, 255)        # light orange
FIRST_YEAR_COLOR = (144, 238, 144)  # light green
MASK_ALPHA = 0.45

DRAW_FRAME_SUMMARY = True
FRAME_SUMMARY_FONT_SCALE = 0.72
FRAME_SUMMARY_FONT_THICKNESS = 2
FRAME_SUMMARY_MARGIN = 20
FRAME_SUMMARY_PADDING = 14
FRAME_SUMMARY_LINE_GAP = 10
FRAME_SUMMARY_BACKGROUND_ALPHA = 0.65
FRAME_SUMMARY_TEXT_COLOR = (255, 255, 255)
FRAME_SUMMARY_BACKGROUND_COLOR = (0, 0, 0)

# Timing.
PROFILE_ACCURATE_GPU_TIMING = True
PROFILE_WARMUP_FRAMES = 5
PROFILE_PRIMARY_STAGES = [
    "male_inference_ms",
    "cane_inference_ms",
    "cutie_tracking_ms",
    "bud_inference_ms",
    "mask_processing_ms",
    "optical_flow_ms",
    "matching_ms",
    "drawing_ms",
]


# ============================================================
# 3) VALIDATE SETTINGS
# ============================================================

if CUTIE_MAX_INTERNAL_SIZE == 0 or CUTIE_MAX_INTERNAL_SIZE < -1:
    raise ValueError("CUTIE_MAX_INTERNAL_SIZE must be -1 or a positive integer.")
if CUTIE_CORRECTION_INTERVAL < 1:
    raise ValueError("CUTIE_CORRECTION_INTERVAL must be at least 1.")
if CUTIE_MIN_SEED_PIXELS < 1:
    raise ValueError("CUTIE_MIN_SEED_PIXELS must be at least 1.")
if CANE_MIN_MASK_AREA < 1:
    raise ValueError("CANE_MIN_MASK_AREA must be at least 1.")
if CUTIE_EMPTY_RESET_FRAMES < 1:
    raise ValueError("CUTIE_EMPTY_RESET_FRAMES must be at least 1.")
if not (0.0 < CANE_CORRIDOR_RESOLUTION_SCALE <= 1.0):
    raise ValueError("CANE_CORRIDOR_RESOLUTION_SCALE must be >0 and <=1.")
if not (0.0 < REGION_FLOW_RESOLUTION_SCALE <= 1.0):
    raise ValueError("REGION_FLOW_RESOLUTION_SCALE must be >0 and <=1.")
if PENDING_CONFIRM_HITS < 1 or PENDING_CONFIRM_WINDOW < 1:
    raise ValueError("Pending-track confirmation settings must be positive.")


# ============================================================
# 4) TIMING HELPERS
# ============================================================


def profile_cuda_sync():
    if PROFILE_ACCURATE_GPU_TIMING and torch.cuda.is_available():
        torch.cuda.synchronize()


def elapsed_ms(start_time):
    return (time.perf_counter() - start_time) * 1000.0


def safe_percentile(values, percentile):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


# ============================================================
# 5) ORIENTATION AND ROI HELPERS
# ============================================================

LANDSCAPE_FRAME_WIDTH = 3840
LANDSCAPE_FRAME_HEIGHT = 2160
PORTRAIT_FRAME_WIDTH = 2160
PORTRAIT_FRAME_HEIGHT = 3840


def normalise_frame_orientation(frame):
    """Return a 3840 x 2160 landscape frame."""
    if frame is None or frame.size == 0:
        raise ValueError("Decoded video frame is empty.")

    frame_height, frame_width = frame.shape[:2]

    if (
        frame_width == PORTRAIT_FRAME_WIDTH
        and frame_height == PORTRAIT_FRAME_HEIGHT
    ):
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    if (
        frame_width == LANDSCAPE_FRAME_WIDTH
        and frame_height == LANDSCAPE_FRAME_HEIGHT
    ):
        return frame

    raise ValueError(
        "Unsupported decoded frame size: "
        f"{frame_width} x {frame_height}. "
        "Step 3 expects either 2160 x 3840 portrait frames or "
        "3840 x 2160 landscape frames."
    )


def parse_roi_coordinates(
    roi_text,
    image_width,
    image_height,
    argument_label="--include_roi",
):
    if roi_text is None or str(roi_text).strip() == "":
        return None

    parts = [part.strip() for part in str(roi_text).split(",")]
    if len(parts) != 4:
        raise ValueError(
            f"{argument_label} must contain exactly four comma-separated "
            "values: x1,y1,x2,y2. Example: 500,200,3300,2000"
        )

    try:
        x1, y1, x2, y2 = [int(round(float(part))) for part in parts]
    except ValueError as error:
        raise ValueError(
            f"{argument_label} values must be numeric. "
            "Example: 500,200,3300,2000"
        ) from error

    if x1 >= x2 or y1 >= y2:
        raise ValueError(
            f"{argument_label} requires x1 < x2 and y1 < y2. "
            f"Received: {x1},{y1},{x2},{y2}"
        )

    clipped_x1 = min(max(0, x1), image_width - 1)
    clipped_y1 = min(max(0, y1), image_height - 1)
    clipped_x2 = min(max(0, x2), image_width - 1)
    clipped_y2 = min(max(0, y2), image_height - 1)

    if clipped_x1 >= clipped_x2 or clipped_y1 >= clipped_y2:
        raise ValueError(
            f"{argument_label} does not overlap the video frame after "
            f"clipping. Video frame: 0,0,{image_width - 1},"
            f"{image_height - 1}. Received: {x1},{y1},{x2},{y2}"
        )

    clipped_roi = (clipped_x1, clipped_y1, clipped_x2, clipped_y2)
    if clipped_roi != (x1, y1, x2, y2):
        print(
            f"{argument_label} was clipped to the video frame:",
            f"{clipped_x1},{clipped_y1},{clipped_x2},{clipped_y2}",
        )
    return clipped_roi


def parse_include_roi_ranges(
    ranges_text,
    image_width,
    image_height,
    video_frame_count=None,
):
    if ranges_text is None or str(ranges_text).strip() == "":
        return []

    parsed_ranges = []
    entries = [entry.strip() for entry in str(ranges_text).split(";")]

    for entry_index, entry in enumerate(entries, start=1):
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                "Each --include_roi_ranges entry must use "
                "'start_frame-end_frame:x1,y1,x2,y2'. "
                f"Invalid entry {entry_index}: {entry!r}"
            )

        frame_range_text, roi_text = entry.split(":", 1)
        frame_parts = [part.strip() for part in frame_range_text.split("-")]
        if len(frame_parts) != 2:
            raise ValueError(
                "Each frame range must contain a start and end frame joined "
                f"by '-'. Invalid entry {entry_index}: {entry!r}"
            )

        try:
            start_frame = int(frame_parts[0])
            end_frame = int(frame_parts[1])
        except ValueError as error:
            raise ValueError(
                "Frame-range values must be whole numbers. "
                f"Invalid entry {entry_index}: {entry!r}"
            ) from error

        if start_frame < 0 or end_frame < 0:
            raise ValueError(
                "Frame-range values cannot be negative. "
                f"Invalid entry {entry_index}: {entry!r}"
            )
        if start_frame > end_frame:
            raise ValueError(
                "Each frame range requires start_frame <= end_frame. "
                f"Invalid entry {entry_index}: {entry!r}"
            )

        if video_frame_count is not None:
            final_video_frame = int(video_frame_count) - 1
            if start_frame > final_video_frame:
                raise ValueError(
                    f"Frame range {start_frame}-{end_frame} starts after the "
                    f"last video frame ({final_video_frame})."
                )
            if end_frame > final_video_frame:
                print(
                    f"Frame range {start_frame}-{end_frame} was clipped to "
                    f"{start_frame}-{final_video_frame}."
                )
                end_frame = final_video_frame

        roi = parse_roi_coordinates(
            roi_text,
            image_width,
            image_height,
            argument_label=(
                f"--include_roi_ranges entry {entry_index} "
                f"(frames {start_frame}-{end_frame})"
            ),
        )

        parsed_ranges.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "roi": roi,
        })

    if not parsed_ranges:
        raise ValueError(
            "--include_roi_ranges was supplied but contained no valid entries."
        )

    parsed_ranges.sort(key=lambda item: item["start_frame"])
    for previous, current in zip(parsed_ranges, parsed_ranges[1:]):
        if current["start_frame"] <= previous["end_frame"]:
            raise ValueError(
                "--include_roi_ranges entries cannot overlap. "
                f"Overlapping ranges: {previous['start_frame']}-"
                f"{previous['end_frame']} and {current['start_frame']}-"
                f"{current['end_frame']}."
            )

    return parsed_ranges


# ============================================================
# 6) CUTIE + BUD TRACKING HELPERS FROM THE TEST PIPELINE
# ============================================================

def get_class_id_by_name(model, class_name):
    """Find the model class ID corresponding to a class name."""
    for class_id, name in model.names.items():
        if str(name).lower() == str(class_name).lower():
            return int(class_id)

    raise ValueError(
        f"Class '{class_name}' was not found.\n"
        f"Available classes: {model.names}"
    )


def make_odd(value):
    value = int(value)
    if value <= 0:
        return 0
    if value % 2 == 0:
        value += 1
    return value


def scaled_odd_size(value, scale, minimum=3):
    """Scale a pixel window while keeping it odd for OpenCV."""
    scaled = max(minimum, int(round(float(value) * float(scale))))
    if scaled % 2 == 0:
        scaled += 1
    return scaled


def add_polygons_to_mask(result, binary_mask, min_area=0):
    """
    Add YOLO segmentation polygons to one binary mask.

    Returns the number of polygons actually added.
    """
    if result is None or result.masks is None:
        return 0

    added = 0

    for polygon in result.masks.xy:
        if polygon is None or len(polygon) < 3:
            continue

        polygon_np = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)

        if min_area > 0:
            area = abs(float(cv2.contourArea(polygon_np)))
            if area < float(min_area):
                continue

        polygon_int = np.rint(polygon_np).astype(np.int32)
        cv2.fillPoly(binary_mask, [polygon_int], 255)
        added += 1

    return added


def clean_yolo_first_year_mask(mask):
    """Small cleanup of the combined current-frame YOLO first-year mask."""
    output = (mask > 0).astype(np.uint8) * 255

    kernel_size = make_odd(CANE_YOLO_CLOSE_KERNEL)

    if kernel_size > 0 and CANE_YOLO_CLOSE_ITER > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        output = cv2.morphologyEx(
            output,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=CANE_YOLO_CLOSE_ITER,
        )

    return output


def frame_to_cutie_tensor(frame_bgr):
    """
    Convert OpenCV BGR uint8 HxWx3 frame to Cutie's RGB float tensor 3xHxW.
    Normalization beyond [0,1] is handled inside Cutie.
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    tensor = torch.from_numpy(
        np.ascontiguousarray(frame_rgb)
    ).permute(2, 0, 1).float().div_(255.0)

    return tensor.cuda(non_blocking=True)


def binary_mask_to_cutie_tensor(binary_mask):
    """
    Convert a binary OpenCV mask to an indexed Cutie mask:
      background = 0
      tracked foreground = CUTIE_OBJECT_ID
    """
    indexed = np.zeros(binary_mask.shape, dtype=np.uint8)
    indexed[binary_mask > 0] = int(CUTIE_OBJECT_ID)

    return torch.from_numpy(indexed).long().cuda(non_blocking=True)


def cutie_output_to_binary(processor, output_prob):
    """Convert Cutie probabilities to a full-resolution binary uint8 mask."""
    indexed = processor.output_prob_to_mask(output_prob)
    indexed_np = indexed.detach().to("cpu").numpy()

    output = np.zeros(indexed_np.shape, dtype=np.uint8)
    output[indexed_np == int(CUTIE_OBJECT_ID)] = 255
    return output


def create_cutie_processor(cutie_model):
    """Create one Cutie InferenceCore for the current video segment."""
    processor = InferenceCore(cutie_model, cfg=cutie_model.cfg)
    processor.max_internal_size = int(CUTIE_MAX_INTERNAL_SIZE)
    return processor


def run_cutie_propagation(processor, image_tensor):
    """Propagate the remembered foreground mask into the current frame."""
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=bool(CUTIE_USE_AMP)):
            output_prob = processor.step(image_tensor)

    return cutie_output_to_binary(processor, output_prob)


def run_cutie_seed(processor, image_tensor, seed_mask):
    """
    Initial Cutie seed.

    This is used only when a new Cutie processor is first created (or after an
    emergency restart). The initial YOLO mask defines object ID 1 for this independent stream.
    """
    mask_tensor = binary_mask_to_cutie_tensor(seed_mask)

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=bool(CUTIE_USE_AMP)):
            output_prob = processor.step(
                image_tensor,
                mask_tensor,
                objects=[int(CUTIE_OBJECT_ID)],
            )

    return cutie_output_to_binary(processor, output_prob)


def prepare_cutie_internal_image(processor, image_tensor):
    """
    Reproduce the image resize/pad used inside InferenceCore.step().

    This lets us reuse the CURRENT frame's cached image features when writing a
    positive-only corrected probability map into Cutie's memory, without
    calling processor.step() a second time.
    """
    internal_image = image_tensor
    h, w = internal_image.shape[-2:]

    if processor.max_internal_size > 0:
        min_side = min(h, w)

        if min_side > processor.max_internal_size:
            new_h = int(h / min_side * processor.max_internal_size)
            new_w = int(w / min_side * processor.max_internal_size)

            internal_image = F.interpolate(
                internal_image.unsqueeze(0),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )[0]

    # InferenceCore.step() has already computed processor.pad for this frame.
    internal_image = F.pad(internal_image, processor.pad)
    internal_image = internal_image.unsqueeze(0)

    if processor.flip_aug:
        internal_image = torch.cat(
            [internal_image, torch.flip(internal_image, dims=[-1])],
            dim=0,
        )

    return internal_image


def prepare_positive_yolo_mask_for_cutie(processor, yolo_binary_mask, image_tensor):
    """
    Convert the current full-resolution YOLO binary mask into the same internal
    spatial size/padding used by Cutie's current frame.

    The returned tensor is boolean with shape:
        1 x 1 x H_internal_padded x W_internal_padded

    It contains ONLY positive YOLO cane evidence. Zero means "no new positive
    evidence"; it does NOT mean "force background".
    """
    positive = torch.from_numpy(
        (yolo_binary_mask > 0).astype(np.float32)
    ).unsqueeze(0).unsqueeze(0)

    positive = positive.to(
        device=image_tensor.device,
        dtype=image_tensor.dtype,
        non_blocking=True,
    )

    h, w = image_tensor.shape[-2:]

    if processor.max_internal_size > 0:
        min_side = min(h, w)

        if min_side > processor.max_internal_size:
            new_h = int(h / min_side * processor.max_internal_size)
            new_w = int(w / min_side * processor.max_internal_size)

            # Nearest-exact preserves the binary character of the YOLO mask
            # while matching Cutie's indexed-mask resize convention.
            positive = F.interpolate(
                positive,
                size=(new_h, new_w),
                mode="nearest-exact",
            )

    positive = F.pad(positive, processor.pad)

    return positive > 0.5


def run_cutie_positive_yolo_correction(
    processor,
    image_tensor,
    yolo_binary_mask,
):
    """
    Propagate Cutie ONCE for the current frame, then add positive YOLO evidence
    to that same frame's Cutie memory.

    Crucially:
      - YOLO-positive pixels can only increase foreground probability.
      - YOLO-zero / missed pixels never reduce Cutie's predicted probability.
      - processor.step() is called only once for this video frame.
      - The SAME InferenceCore and its accumulated memory are retained.

    We use Cutie's current foreground probability (processor.last_mask) as the
    starting point, set YOLO-positive pixels to probability 1.0, and then write
    that corrected probability into the current processor memory using the
    frame features already computed by InferenceCore.step().
    """
    # --------------------------------------------------------
    # 1. Normal Cutie prediction for this frame.
    #
    # end=True prevents the uncorrected prediction from being automatically
    # written into memory on this step. delete_buffer=False keeps the current
    # image features cached so we can add the corrected version ourselves.
    # --------------------------------------------------------
    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=bool(CUTIE_USE_AMP)):
            output_prob = processor.step(
                image_tensor,
                end=True,
                delete_buffer=False,
            )

    cutie_binary = cutie_output_to_binary(processor, output_prob)

    # --------------------------------------------------------
    # 2. Start from Cutie's OWN soft foreground probability.
    #    This is the key positive-only behavior.
    # --------------------------------------------------------
    corrected_prob = processor.last_mask.detach().clone()

    if corrected_prob.shape[1] < 1:
        # Defensive fallback; should not happen once object ID 1 is initialized.
        processor.image_feature_store.delete(processor.curr_ti)
        return cutie_binary

    # --------------------------------------------------------
    # 3. Map current YOLO-positive pixels into Cutie's internal resolution.
    # --------------------------------------------------------
    positive_yolo = prepare_positive_yolo_mask_for_cutie(
        processor,
        yolo_binary_mask,
        image_tensor,
    )

    if processor.flip_aug:
        positive_yolo = torch.cat(
            [positive_yolo, torch.flip(positive_yolo, dims=[-1])],
            dim=0,
        )

    # --------------------------------------------------------
    # 4. POSITIVE-ONLY correction.
    #
    # Wherever YOLO says "cane", force Cutie's first foreground object toward
    # certainty. Everywhere else remains exactly as Cutie predicted it.
    # --------------------------------------------------------
    corrected_prob[:, 0:1] = torch.where(
        positive_yolo,
        torch.ones_like(corrected_prob[:, 0:1]),
        corrected_prob[:, 0:1],
    )

    corrected_prob = corrected_prob.clamp_(0.0, 1.0)

    # Keep the corrected probability as Cutie's current last mask.
    processor.last_mask = corrected_prob

    # --------------------------------------------------------
    # 5. Add the corrected CURRENT frame to the SAME Cutie memory.
    #
    # We intentionally reuse Cutie's cached features for processor.curr_ti.
    # This avoids calling processor.step() twice for one physical video frame.
    # --------------------------------------------------------
    internal_image = prepare_cutie_internal_image(
        processor,
        image_tensor,
    )

    _, pix_feat = processor.image_feature_store.get_features(
        processor.curr_ti,
        internal_image,
    )

    key, shrinkage, selection = processor.image_feature_store.get_key(
        processor.curr_ti,
        internal_image,
    )

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=bool(CUTIE_USE_AMP)):
            processor._add_memory(
                internal_image,
                pix_feat,
                corrected_prob,
                key,
                shrinkage,
                selection,
                is_deep_update=True,
                force_permanent=False,
            )

    # We kept the feature buffer only long enough to perform the correction.
    processor.image_feature_store.delete(processor.curr_ti)

    return cutie_binary


def fuse_cutie_and_yolo_masks(cutie_mask, yolo_mask):
    """
    Fuse temporal Cutie continuity with current YOLO evidence.

    Cutie provides continuity through brief YOLO misses.
    YOLO adds currently visible foreground pixels immediately.
    """
    if cutie_mask is None:
        fused = yolo_mask.copy()
    elif yolo_mask is None:
        fused = cutie_mask.copy()
    else:
        fused = cv2.bitwise_or(cutie_mask, yolo_mask)

    kernel_size = make_odd(CUTIE_FUSION_CLOSE_KERNEL)

    if kernel_size > 0 and CUTIE_FUSION_CLOSE_ITER > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        fused = cv2.morphologyEx(
            fused,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=CUTIE_FUSION_CLOSE_ITER,
        )

    return (fused > 0).astype(np.uint8) * 255


def make_cutie_stream_state(name):
    """Create independent temporal state and diagnostics for one Cutie stream."""
    return {
        "name": str(name),
        "processor": None,
        "initialized": False,
        "last_correction_frame": None,
        "empty_streak": 0,
        "seed_count": 0,
        "propagation_frames": 0,
        "correction_count": 0,
        "reset_count": 0,
        "yolo_nonempty_frames": 0,
        "cutie_nonempty_frames": 0,
        "fused_nonempty_frames": 0,
    }


def update_cutie_stream(
    state,
    cutie_model,
    image_tensor,
    yolo_mask,
    frame_idx,
    frame_height,
    frame_width,
):
    """
    Advance ONE independent Cutie stream by one physical video frame.

    Behavior:
      - Initial reliable YOLO mask seeds this stream's own InferenceCore.
      - The same processor then propagates continuously.
      - Every CUTIE_CORRECTION_INTERVAL frames, positive YOLO pixels can
        strengthen/add foreground in memory.
      - YOLO-zero pixels never force remembered foreground to background.
      - A new processor is created only after this stream has been empty for
        CUTIE_EMPTY_RESET_FRAMES consecutive frames.
    """
    yolo_pixels = int(np.count_nonzero(yolo_mask))
    yolo_valid = yolo_pixels >= CUTIE_MIN_SEED_PIXELS

    if yolo_valid:
        state["yolo_nonempty_frames"] += 1

    # --------------------------------------------------------
    # Initial seed / post-reset seed.
    # --------------------------------------------------------
    if not state["initialized"]:
        if yolo_valid:
            state["processor"] = create_cutie_processor(cutie_model)

            cutie_mask = run_cutie_seed(
                state["processor"],
                image_tensor,
                yolo_mask,
            )

            output_mask = fuse_cutie_and_yolo_masks(
                cutie_mask,
                yolo_mask,
            )

            state["initialized"] = True
            state["last_correction_frame"] = frame_idx
            state["empty_streak"] = 0
            state["seed_count"] += 1
        else:
            # No reliable foreground exists yet for this stream.
            cutie_mask = np.zeros(
                (frame_height, frame_width),
                dtype=np.uint8,
            )
            output_mask = yolo_mask.copy()

    # --------------------------------------------------------
    # Continuous propagation / positive-only correction.
    # --------------------------------------------------------
    else:
        correction_due = (
            state["last_correction_frame"] is None
            or (
                frame_idx - state["last_correction_frame"]
                >= CUTIE_CORRECTION_INTERVAL
            )
        )

        apply_positive_correction = correction_due and yolo_valid

        if apply_positive_correction:
            # processor.step() is called exactly ONCE inside this helper.
            # Only YOLO-positive pixels strengthen the current Cutie memory.
            cutie_mask = run_cutie_positive_yolo_correction(
                state["processor"],
                image_tensor,
                yolo_mask,
            )

            state["last_correction_frame"] = frame_idx
            state["correction_count"] += 1
        else:
            cutie_mask = run_cutie_propagation(
                state["processor"],
                image_tensor,
            )

        state["propagation_frames"] += 1

        cutie_pixels = int(np.count_nonzero(cutie_mask))

        if cutie_pixels >= CUTIE_MIN_SEED_PIXELS:
            state["cutie_nonempty_frames"] += 1
            state["empty_streak"] = 0
        else:
            state["empty_streak"] += 1

        # Visible output remains inclusive: current YOLO detections appear
        # immediately even between memory-correction frames.
        output_mask = fuse_cutie_and_yolo_masks(
            cutie_mask,
            yolo_mask,
        )

        # ----------------------------------------------------
        # Emergency reset affects ONLY this independent stream.
        # ----------------------------------------------------
        if state["empty_streak"] >= CUTIE_EMPTY_RESET_FRAMES:
            state["processor"] = None
            state["initialized"] = False
            state["last_correction_frame"] = None
            state["empty_streak"] = 0
            state["reset_count"] += 1

            # If YOLO is valid on this same frame, restart immediately.
            if yolo_valid:
                state["processor"] = create_cutie_processor(cutie_model)

                cutie_mask = run_cutie_seed(
                    state["processor"],
                    image_tensor,
                    yolo_mask,
                )

                output_mask = fuse_cutie_and_yolo_masks(
                    cutie_mask,
                    yolo_mask,
                )

                state["initialized"] = True
                state["last_correction_frame"] = frame_idx
                state["seed_count"] += 1

    if np.count_nonzero(output_mask) >= CUTIE_MIN_SEED_PIXELS:
        state["fused_nonempty_frames"] += 1

    return output_mask


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))


def bbox_center(box):
    x1, y1, x2, y2 = box
    return (
        0.5 * (float(x1) + float(x2)),
        0.5 * (float(y1) + float(y2))
    )


def point_is_inside_include_roi(point, include_roi):
    """Return True when a point is inside the active counting ROI."""
    if include_roi is None:
        return True

    x, y = map(float, point)
    roi_x1, roi_y1, roi_x2, roi_y2 = include_roi

    return (
        roi_x1 <= x <= roi_x2
        and roi_y1 <= y <= roi_y2
    )


def box_center_is_inside_include_roi(box, include_roi):
    """Apply the active include rule using the detection-box centre."""
    return point_is_inside_include_roi(
        bbox_center(box),
        include_roi,
    )


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, float(ix2 - ix1))
    ih = max(0.0, float(iy2 - iy1))
    intersection = iw * ih

    union = box_area(box_a) + box_area(box_b) - intersection + 1e-6
    return intersection / union


def box_area_ratio(box_a, box_b):
    area_a = max(1.0, box_area(box_a))
    area_b = max(1.0, box_area(box_b))
    return max(area_a / area_b, area_b / area_a)


def center_distance(box_a, box_b):
    ax, ay = bbox_center(box_a)
    bx, by = bbox_center(box_b)
    return math.hypot(ax - bx, ay - by)


def expand_box(box, expand, image_width, image_height):
    x1, y1, x2, y2 = box
    return [
        max(0, int(round(x1 - expand))),
        max(0, int(round(y1 - expand))),
        min(image_width - 1, int(round(x2 + expand))),
        min(image_height - 1, int(round(y2 + expand)))
    ]


def shift_box(box, dx, dy, image_width, image_height):
    x1, y1, x2, y2 = box
    width = max(1, int(round(x2 - x1)))
    height = max(1, int(round(y2 - y1)))

    nx1 = int(round(x1 + dx))
    ny1 = int(round(y1 + dy))

    nx1 = min(max(0, nx1), max(0, image_width - 1 - width))
    ny1 = min(max(0, ny1), max(0, image_height - 1 - height))

    return [nx1, ny1, nx1 + width, ny1 + height]


def clip_point(point, image_width, image_height):
    x, y = point
    return (
        float(min(max(0.0, x), image_width - 1.0)),
        float(min(max(0.0, y), image_height - 1.0))
    )


def remove_duplicate_buds_same_frame(detections):
    """Conservative same-frame duplicate filtering."""
    if not detections:
        return []

    sorted_detections = sorted(
        detections,
        key=lambda item: item["conf"],
        reverse=True
    )

    kept = []

    for detection in sorted_detections:
        duplicate = False

        for existing in kept:
            overlap = box_iou(detection["box"], existing["box"])
            distance = center_distance(detection["box"], existing["box"])

            if overlap >= SAME_FRAME_DUP_STRONG_IOU:
                duplicate = True
                break

            if (
                overlap >= SAME_FRAME_DUP_WEAK_IOU
                and distance <= SAME_FRAME_DUP_CENTER_DIST
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(detection)

    return kept


def estimate_global_affine(old_points, new_points):
    identity = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32
    )

    if old_points is None or new_points is None or len(old_points) < 3:
        return identity

    old_points = np.asarray(old_points, dtype=np.float32).reshape(-1, 2)
    new_points = np.asarray(new_points, dtype=np.float32).reshape(-1, 2)

    affine, inliers = cv2.estimateAffinePartial2D(
        old_points,
        new_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=GLOBAL_AFFINE_RANSAC_THRESHOLD,
        maxIters=2000,
        confidence=0.99,
        refineIters=10
    )

    if affine is not None and np.isfinite(affine).all():
        return affine.astype(np.float32)

    displacement = np.median(new_points - old_points, axis=0)
    identity[0, 2] = float(displacement[0])
    identity[1, 2] = float(displacement[1])
    return identity


def create_acceptance_mask(stabilised_mask):
    """Create the small bud-acceptance expansion."""
    if CANE_ACCEPT_DILATE_KERNEL <= 0 or CANE_ACCEPT_DILATE_ITER <= 0:
        result = stabilised_mask.copy()
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CANE_ACCEPT_DILATE_KERNEL, CANE_ACCEPT_DILATE_KERNEL))
        result = cv2.dilate(stabilised_mask, kernel, iterations=CANE_ACCEPT_DILATE_ITER)
    return result


def create_cane_corridor(cane_mask):
    """
    Create the wide regional optical-flow corridor at reduced resolution.

    In the frame loop below, the source is the continuous Cutie-established
    countable first-year cane mask. The corridor is expanded by the configured
    cane-corridor radius before regional optical flow is measured.
    """
    image_height, image_width = cane_mask.shape[:2]
    if (image_width, image_height) == (CANE_CORRIDOR_W, CANE_CORRIDOR_H):
        reduced_mask = cane_mask.copy()
    else:
        reduced_mask = cv2.resize(cane_mask, (CANE_CORRIDOR_W, CANE_CORRIDOR_H), interpolation=cv2.INTER_NEAREST)
    kernel_width = 2 * CANE_CORRIDOR_RADIUS_X_SCALED + 1
    kernel_height = 2 * CANE_CORRIDOR_RADIUS_Y_SCALED + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_width, kernel_height))
    result = cv2.dilate(reduced_mask, kernel, iterations=1)
    return result


def moving_average_points(points, window):
    points = np.asarray(points, dtype=np.float32)
    if len(points) < 3 or window <= 1:
        return points

    radius = window // 2
    smoothed = []

    for index in range(len(points)):
        start = max(0, index - radius)
        end = min(len(points), index + radius + 1)
        smoothed.append(np.mean(points[start:end], axis=0))

    return np.asarray(smoothed, dtype=np.float32)


def build_cane_structures(stabilised_mask):
    """Build cane centrelines with ROI extraction and one-pass bin grouping."""
    working = stabilised_mask.copy()
    if CANE_COMPONENT_CLOSE_KERNEL > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CANE_COMPONENT_CLOSE_KERNEL, CANE_COMPONENT_CLOSE_KERNEL))
        working = cv2.morphologyEx(working, cv2.MORPH_CLOSE, kernel)
    working_binary = (working > 0).astype(np.uint8)
    number_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            working_binary,
            connectivity=8
        )
    )
    retained_label_ids = []
    for label_id in range(1, number_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area >= CANE_COMPONENT_MIN_AREA:
            retained_label_ids.append(label_id)
    structures = []
    cane_id = 1
    for label_id in retained_label_ids:
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        component_labels = labels[y:y + height, x:x + width]
        local_ys, local_xs = np.where(component_labels == label_id)
        ys = local_ys + y
        xs = local_xs + x
        if len(xs) < 10:
            continue
        points = np.column_stack([xs, ys]).astype(np.float32)
        if len(points) > 12000:
            step = max(1, len(points) // 12000)
            pca_points = points[::step]
        else:
            pca_points = points
        centroid = np.mean(pca_points, axis=0)
        centred = pca_points - centroid
        covariance = np.cov(centred.T)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            main_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        except np.linalg.LinAlgError:
            main_axis = np.asarray([1.0, 0.0], dtype=np.float32)
        main_axis = np.asarray(main_axis, dtype=np.float32)
        axis_norm = float(np.linalg.norm(main_axis))
        if axis_norm < 1e-06:
            main_axis = np.asarray([1.0, 0.0], dtype=np.float32)
        else:
            main_axis /= axis_norm
        all_t = (points - centroid) @ main_axis
        t_min = float(np.min(all_t))
        t_max = float(np.max(all_t))
        if t_max - t_min < CENTERLINE_BIN_SIZE:
            line_points = np.asarray([centroid], dtype=np.float32)
        else:
            bin_edges = np.arange(t_min, t_max + CENTERLINE_BIN_SIZE, CENTERLINE_BIN_SIZE, dtype=np.float32)
            number_bins = max(0, len(bin_edges) - 1)
            bin_ids = (np.searchsorted(bin_edges, all_t, side='right') - 1).astype(np.int32, copy=False)
            valid_bin_points = (bin_ids >= 0) & (bin_ids < number_bins)
            valid_bin_ids = bin_ids[valid_bin_points]
            valid_points = points[valid_bin_points]
            if len(valid_bin_ids) > 0:
                sort_order = np.argsort(valid_bin_ids, kind='stable')
                sorted_bin_ids = valid_bin_ids[sort_order]
                sorted_points = valid_points[sort_order]
                bin_counts = np.bincount(sorted_bin_ids, minlength=number_bins)
                bin_offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(bin_counts, dtype=np.int64)])
            else:
                sorted_points = np.empty((0, 2), dtype=np.float32)
                bin_counts = np.zeros(number_bins, dtype=np.int64)
                bin_offsets = np.zeros(number_bins + 1, dtype=np.int64)
            line_points_list = []
            for bin_index in range(number_bins):
                points_in_bin = int(bin_counts[bin_index])
                if points_in_bin < CENTERLINE_MIN_POINTS_PER_BIN:
                    continue
                group_start = int(bin_offsets[bin_index])
                group_end = int(bin_offsets[bin_index + 1])
                line_points_list.append(np.median(sorted_points[group_start:group_end], axis=0))
            if len(line_points_list) < 2:
                line_points = np.asarray([centroid + t_min * main_axis, centroid + t_max * main_axis], dtype=np.float32)
            else:
                line_points = np.asarray(line_points_list, dtype=np.float32)
        line_points = moving_average_points(line_points, CENTERLINE_SMOOTH_WINDOW)
        if len(line_points) == 1:
            cumulative = np.asarray([0.0], dtype=np.float32)
            total_length = 1.0
        else:
            segment_lengths = np.linalg.norm(np.diff(line_points, axis=0), axis=1)
            cumulative = np.concatenate([np.asarray([0.0], dtype=np.float32), np.cumsum(segment_lengths)]).astype(np.float32)
            total_length = max(1.0, float(cumulative[-1]))
        structures.append({'cane_id': cane_id, 'label_id': label_id, 'area': area, 'centroid': tuple(map(float, centroids[label_id])), 'bbox': [x, y, x + width, y + height], 'line_points': line_points, 'cumulative_length': cumulative, 'total_length': total_length})
        cane_id += 1
    return structures


def assign_point_to_cane(point, cane_structures):
    if not cane_structures:
        return None, None, None

    px, py = map(float, point)
    query = np.asarray([px, py], dtype=np.float32)

    best = None

    for structure in cane_structures:
        x1, y1, x2, y2 = structure["bbox"]
        margin = CANE_ASSIGN_MAX_DISTANCE

        if (
            px < x1 - margin
            or px > x2 + margin
            or py < y1 - margin
            or py > y2 + margin
        ):
            continue

        line_points = structure["line_points"]
        distances = np.linalg.norm(line_points - query, axis=1)
        index = int(np.argmin(distances))
        distance = float(distances[index])

        if best is None or distance < best[0]:
            order = float(
                structure["cumulative_length"][index]
                / structure["total_length"]
            )
            best = (distance, structure["cane_id"], order)

    if best is None or best[0] > CANE_ASSIGN_MAX_DISTANCE:
        return None, None, None

    return best[1], best[2], best[0]


def empty_motion_map(image_width, image_height):
    return {
        "vectors": np.zeros(
            (REGION_GRID_ROWS, REGION_GRID_COLS, 2),
            dtype=np.float32
        ),
        "reliability": np.zeros(
            (REGION_GRID_ROWS, REGION_GRID_COLS),
            dtype=np.float32
        ),
        "image_width": image_width,
        "image_height": image_height,
        "lk_dx_px": 0.0,
        "lk_dy_px": 0.0,
        "lk_shift_px": 0.0,
        "lk_tracks_used": 0,
        "lk_ok": False,
        "affine": np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32
        )
    }


def estimate_regional_motion(previous_gray, current_gray, previous_corridor_mask, image_width, image_height):
    """
    Estimate regional motion using reduced-resolution images.

    Feature detection and forward/backward LK run in the reduced regional
    coordinate system. Valid points are then converted back to full-resolution
    coordinates before affine estimation, grid assignment and motion-vector
    construction. The returned motion map therefore remains fully compatible
    with the existing full-resolution tracking and mask-stabilisation logic.
    """
    motion_map = empty_motion_map(image_width, image_height)
    if previous_gray is None or previous_corridor_mask is None:
        return motion_map
    flow_height, flow_width = previous_gray.shape[:2]
    if current_gray.shape[:2] != (flow_height, flow_width) or previous_corridor_mask.shape[:2] != (flow_height, flow_width):
        raise ValueError('Regional optical-flow inputs must have matching dimensions.')
    if flow_width != REGION_FLOW_W or flow_height != REGION_FLOW_H:
        raise ValueError(f'Regional flow received unexpected image dimensions: {flow_width}x{flow_height}; expected {REGION_FLOW_W}x{REGION_FLOW_H}.')
    points = cv2.goodFeaturesToTrack(previous_gray, maxCorners=REGION_MAX_CORNERS, qualityLevel=REGION_QUALITY_LEVEL, minDistance=REGION_MIN_DISTANCE_SCALED, mask=previous_corridor_mask, blockSize=REGION_BLOCK_SIZE_SCALED, useHarrisDetector=False)
    if points is None or len(points) < 3:
        return motion_map
    lk_params = {'winSize': (REGION_LK_WIN_SIZE_SCALED, REGION_LK_WIN_SIZE_SCALED), 'maxLevel': REGION_LK_MAX_LEVEL, 'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
    new_points, status_forward, _ = cv2.calcOpticalFlowPyrLK(previous_gray, current_gray, points, None, **lk_params)
    if new_points is None or status_forward is None:
        return motion_map
    returned_points, status_backward, _ = cv2.calcOpticalFlowPyrLK(current_gray, previous_gray, new_points, None, **lk_params)
    if returned_points is None or status_backward is None:
        return motion_map
    old_flat_reduced = points.reshape(-1, 2)
    new_flat_reduced = new_points.reshape(-1, 2)
    returned_flat_reduced = returned_points.reshape(-1, 2)
    fb_error_reduced = np.linalg.norm(returned_flat_reduced - old_flat_reduced, axis=1)
    displacement_reduced = new_flat_reduced - old_flat_reduced
    displacement_norm_reduced = np.linalg.norm(displacement_reduced, axis=1)
    valid = status_forward.reshape(-1).astype(bool) & status_backward.reshape(-1).astype(bool) & np.isfinite(new_flat_reduced).all(axis=1) & np.isfinite(returned_flat_reduced).all(axis=1) & (fb_error_reduced <= REGION_FB_MAX_ERROR_SCALED) & (displacement_norm_reduced <= REGION_MAX_REASONABLE_DISPLACEMENT_SCALED) & (new_flat_reduced[:, 0] >= 0) & (new_flat_reduced[:, 0] < flow_width) & (new_flat_reduced[:, 1] >= 0) & (new_flat_reduced[:, 1] < flow_height)
    old_valid_reduced = old_flat_reduced[valid]
    new_valid_reduced = new_flat_reduced[valid]
    old_valid = old_valid_reduced.astype(np.float32, copy=True)
    new_valid = new_valid_reduced.astype(np.float32, copy=True)
    if len(old_valid) > 0:
        old_valid[:, 0] /= REGION_FLOW_SCALE_X
        old_valid[:, 1] /= REGION_FLOW_SCALE_Y
        new_valid[:, 0] /= REGION_FLOW_SCALE_X
        new_valid[:, 1] /= REGION_FLOW_SCALE_Y
    if len(old_valid) > 0:
        global_displacements = new_valid - old_valid
        global_median = np.median(global_displacements, axis=0)
        motion_map['lk_dx_px'] = float(global_median[0])
        motion_map['lk_dy_px'] = float(global_median[1])
        motion_map['lk_shift_px'] = float(np.linalg.norm(global_median))
        motion_map['lk_tracks_used'] = int(len(old_valid))
        motion_map['lk_ok'] = bool(len(old_valid) >= 3)

    motion_map['affine'] = estimate_global_affine(old_valid, new_valid)
    if len(old_valid) < 3:
        return motion_map
    vectors = motion_map['vectors']
    reliability = motion_map['reliability']
    measured = np.zeros((REGION_GRID_ROWS, REGION_GRID_COLS), dtype=bool)
    cell_width = image_width / float(REGION_GRID_COLS)
    cell_height = image_height / float(REGION_GRID_ROWS)
    displacements = new_valid - old_valid

    # Assign every valid regional-flow point to a grid cell once.
    #
    # The previous implementation rebuilt a full boolean mask over every
    # valid point for each of the 48 grid cells. Here, searchsorted applies
    # the same half-open interval rule once:
    #
    #     cell_start <= coordinate < cell_end
    #
    # Points exactly on an internal grid boundary therefore enter the cell
    # to the right/below, matching the original comparisons. Points on or
    # outside the final image boundary are excluded.
    # NumPy compares the float32 point arrays with the scalar loop
    # boundaries at float32 precision. Constructing the edges with the same
    # final cast preserves that exact boundary behaviour.
    x_edges = np.asarray(
        [index * cell_width for index in range(REGION_GRID_COLS + 1)],
        dtype=np.float32
    )
    y_edges = np.asarray(
        [index * cell_height for index in range(REGION_GRID_ROWS + 1)],
        dtype=np.float32
    )

    point_cols = (
        np.searchsorted(x_edges, old_valid[:, 0], side='right') - 1
    ).astype(np.int32, copy=False)
    point_rows = (
        np.searchsorted(y_edges, old_valid[:, 1], side='right') - 1
    ).astype(np.int32, copy=False)

    point_in_grid = (
        (point_cols >= 0)
        & (point_cols < REGION_GRID_COLS)
        & (point_rows >= 0)
        & (point_rows < REGION_GRID_ROWS)
    )

    valid_cell_ids = (
        point_rows[point_in_grid] * REGION_GRID_COLS
        + point_cols[point_in_grid]
    ).astype(np.int32, copy=False)
    valid_cell_displacements = displacements[point_in_grid]

    number_grid_cells = REGION_GRID_ROWS * REGION_GRID_COLS

    if len(valid_cell_ids) > 0:
        # Stable sorting preserves the original point order inside each cell.
        # This keeps the existing [::step] point limiting exactly equivalent.
        cell_sort_order = np.argsort(valid_cell_ids, kind='stable')
        sorted_cell_ids = valid_cell_ids[cell_sort_order]
        sorted_displacements = valid_cell_displacements[cell_sort_order]

        cell_counts = np.bincount(
            sorted_cell_ids,
            minlength=number_grid_cells
        )
        cell_offsets = np.concatenate(
            [
                np.asarray([0], dtype=np.int64),
                np.cumsum(cell_counts, dtype=np.int64)
            ]
        )
    else:
        sorted_displacements = np.empty((0, 2), dtype=np.float32)
        cell_counts = np.zeros(number_grid_cells, dtype=np.int64)
        cell_offsets = np.zeros(number_grid_cells + 1, dtype=np.int64)

    for row in range(REGION_GRID_ROWS):
        for col in range(REGION_GRID_COLS):
            cell_id = row * REGION_GRID_COLS + col
            cell_count = int(cell_counts[cell_id])

            if cell_count <= 0:
                continue

            cell_start = int(cell_offsets[cell_id])
            cell_end = int(cell_offsets[cell_id + 1])
            cell_displacements = sorted_displacements[cell_start:cell_end]

            if len(cell_displacements) > REGION_MAX_POINTS_PER_CELL:
                step = max(
                    1,
                    len(cell_displacements) // REGION_MAX_POINTS_PER_CELL
                )
                cell_displacements = cell_displacements[::step]

            if len(cell_displacements) < REGION_MIN_POINTS_PER_CELL:
                continue

            median = np.median(cell_displacements, axis=0)
            residual = np.linalg.norm(cell_displacements - median, axis=1)
            mad = float(
                np.median(
                    np.abs(residual - np.median(residual))
                )
            )
            robust_limit = max(
                2.0,
                REGION_OUTLIER_MAD_MULT * 1.4826 * mad
            )
            robust = cell_displacements[residual <= robust_limit]

            if len(robust) >= REGION_MIN_POINTS_PER_CELL:
                vector = np.median(robust, axis=0)
                vectors[row, col] = vector.astype(np.float32)
                measured[row, col] = True
                reliability[row, col] = min(
                    1.0,
                    len(robust) / float(REGION_MIN_POINTS_PER_CELL * 3)
                )
    filled_vectors = vectors.copy()
    filled_reliability = reliability.copy()
    for row in range(REGION_GRID_ROWS):
        for col in range(REGION_GRID_COLS):
            if measured[row, col]:
                continue
            neighbours = []
            for radius in range(1, REGION_FILL_NEIGHBOUR_RADIUS + 1):
                for rr in range(max(0, row - radius), min(REGION_GRID_ROWS, row + radius + 1)):
                    for cc in range(max(0, col - radius), min(REGION_GRID_COLS, col + radius + 1)):
                        if not measured[rr, cc]:
                            continue
                        distance = math.hypot(rr - row, cc - col)
                        if distance <= 0:
                            continue
                        weight = reliability[rr, cc] / distance
                        neighbours.append((weight, vectors[rr, cc]))
                if neighbours:
                    break
            if neighbours:
                weights = np.asarray([item[0] for item in neighbours], dtype=np.float32)
                neighbour_vectors = np.asarray([item[1] for item in neighbours], dtype=np.float32)
                filled_vectors[row, col] = np.average(neighbour_vectors, axis=0, weights=weights)
                filled_reliability[row, col] = min(0.45, float(np.mean(weights)))
    smoothed_vectors = filled_vectors.copy()
    smoothed_reliability = filled_reliability.copy()
    for row in range(REGION_GRID_ROWS):
        for col in range(REGION_GRID_COLS):
            local_vectors = []
            local_weights = []
            for rr in range(max(0, row - 1), min(REGION_GRID_ROWS, row + 2)):
                for cc in range(max(0, col - 1), min(REGION_GRID_COLS, col + 2)):
                    rel = filled_reliability[rr, cc]
                    if rel <= 0:
                        continue
                    spatial = 1.0 / (1.0 + math.hypot(rr - row, cc - col))
                    local_vectors.append(filled_vectors[rr, cc])
                    local_weights.append(rel * spatial)
            if local_vectors:
                smoothed_vectors[row, col] = np.average(np.asarray(local_vectors), axis=0, weights=np.asarray(local_weights))
                smoothed_reliability[row, col] = min(1.0, float(np.max(local_weights)))
    motion_map['vectors'] = smoothed_vectors.astype(np.float32)
    motion_map['reliability'] = smoothed_reliability.astype(np.float32)
    return motion_map


def sample_regional_motion(motion_map, point):
    vectors = motion_map['vectors']
    reliability = motion_map['reliability']
    image_width = motion_map['image_width']
    image_height = motion_map['image_height']
    x, y = map(float, point)
    cell_width = image_width / float(REGION_GRID_COLS)
    cell_height = image_height / float(REGION_GRID_ROWS)
    candidates = []
    for row in range(REGION_GRID_ROWS):
        for col in range(REGION_GRID_COLS):
            rel = float(reliability[row, col])
            if rel <= 0:
                continue
            center_x = (col + 0.5) * cell_width
            center_y = (row + 0.5) * cell_height
            distance = math.hypot(center_x - x, center_y - y)
            candidates.append((distance, rel, vectors[row, col]))
    if not candidates:
        return (np.asarray([0.0, 0.0], dtype=np.float32), 0.0)
    candidates.sort(key=lambda item: item[0])
    candidates = candidates[:4]
    weights = []
    selected_vectors = []
    for distance, rel, vector in candidates:
        weight = rel / max(20.0, distance)
        weights.append(weight)
        selected_vectors.append(vector)
    vector = np.average(np.asarray(selected_vectors, dtype=np.float32), axis=0, weights=np.asarray(weights, dtype=np.float32))
    resulting_reliability = min(1.0, float(np.sum(weights)))
    return (vector.astype(np.float32), resulting_reliability)


def extract_patch(gray_frame, box, image_width, image_height):
    x1, y1, x2, y2 = expand_box(
        box,
        PATCH_EXPAND,
        image_width,
        image_height
    )

    patch = gray_frame[y1:y2 + 1, x1:x2 + 1]
    if patch.size == 0:
        return None

    patch = cv2.resize(
        patch,
        (PATCH_SIZE, PATCH_SIZE),
        interpolation=cv2.INTER_AREA
    )
    return patch.astype(np.float32)


def patch_similarity(patch_a, patch_b):
    if patch_a is None or patch_b is None:
        return 0.0

    if patch_a.shape != patch_b.shape:
        patch_b = cv2.resize(
            patch_b,
            (patch_a.shape[1], patch_a.shape[0])
        )

    a = patch_a.astype(np.float32)
    b = patch_b.astype(np.float32)

    a -= float(np.mean(a))
    b -= float(np.mean(b))

    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator < 1e-6:
        return 0.0

    similarity = float(np.sum(a * b) / denominator)
    return max(-1.0, min(1.0, similarity))


def bud_is_on_cane_mask(
    bud_box,
    acceptance_mask,
    image_width,
    image_height
):
    x1, y1, x2, y2 = expand_box(
        bud_box,
        BUD_BOX_EXPAND_FOR_MASK,
        image_width,
        image_height
    )

    if x2 <= x1 or y2 <= y1:
        return False, 0.0, False

    center_x, center_y = bbox_center([x1, y1, x2, y2])
    center_x_int = int(round(center_x))
    center_y_int = int(round(center_y))

    center_inside = False
    if (
        0 <= center_x_int < image_width
        and 0 <= center_y_int < image_height
    ):
        center_inside = acceptance_mask[center_y_int, center_x_int] > 0

    crop = acceptance_mask[y1:y2 + 1, x1:x2 + 1]
    overlap_ratio = (
        float(np.count_nonzero(crop > 0)) / float(crop.size)
        if crop.size > 0
        else 0.0
    )

    accepted = False
    if USE_BUD_CENTER_INSIDE_MASK and center_inside:
        accepted = True
    if overlap_ratio >= MIN_BUD_BOX_MASK_OVERLAP:
        accepted = True

    return accepted, overlap_ratio, center_inside


def run_bud_inference(frame):
    """Run only the YOLO bud model. Post-processing is timed separately."""
    with torch.inference_mode():
        return bud_model.predict(
            source=frame,
            conf=BUD_CONF_THRES,
            iou=BUD_IOU_THRES,
            imgsz=BUD_IMG_SIZE,
            classes=[BUD_CLASS_ID],
            agnostic_nms=True,
            max_det=BUD_MAX_DET,
            device=DEVICE,
            verbose=False
        )


def postprocess_bud_results(
    results,
    gray_frame,
    acceptance_mask,
    cane_structures,
    image_width,
    image_height,
    include_roi,
    counting_enabled=True,
):
    """
    Convert YOLO results into the detection dictionaries used downstream.

    Same-frame duplicates are removed immediately after extracting only the
    inexpensive box, confidence and class fields. Cane-mask overlap, cane
    assignment and patch extraction are then calculated only for detections
    that survive deduplication. The existing duplicate thresholds, confidence
    ordering and final detection fields remain unchanged.
    """
    if not counting_enabled:
        return []

    if not results or results[0].boxes is None:
        return []

    boxes_object = results[0].boxes
    boxes = boxes_object.xyxy.cpu().numpy()
    confidences = boxes_object.conf.cpu().numpy()
    class_ids = boxes_object.cls.cpu().numpy().astype(int)

    # Build lightweight detections first. The existing deduplication function
    # uses only ``box`` and ``conf``, so expensive per-detection work is not
    # needed until after duplicates have been discarded.
    lightweight_detections = []

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        class_id = int(class_id)
        if class_id != BUD_CLASS_ID:
            continue

        x1, y1, x2, y2 = map(int, box.tolist())
        bud_box = [x1, y1, x2, y2]

        # Optional ROI check retained from the old bud detector.
        # This combined version calls it with include_roi=None.
        if not box_center_is_inside_include_roi(bud_box, include_roi):
            continue

        lightweight_detections.append({
            "box": [x1, y1, x2, y2],
            "conf": float(confidence),
            "class_id": class_id
        })

    # This preserves the previous behaviour: candidates are sorted by
    # confidence and the same IoU/centre-distance rules select the survivors.
    retained_detections = remove_duplicate_buds_same_frame(
        lightweight_detections
    )

    detections = []

    for detection in retained_detections:
        bud_box = detection["box"]

        cane_pass, overlap_ratio, center_inside = bud_is_on_cane_mask(
            bud_box,
            acceptance_mask,
            image_width,
            image_height
        )

        center = bbox_center(bud_box)
        cane_id, cane_order, cane_distance = assign_point_to_cane(
            center,
            cane_structures
        )

        detections.append({
            "box": bud_box,
            "center": center,
            "conf": detection["conf"],
            "class_id": detection["class_id"],
            "cane_pass": bool(cane_pass),
            "mask_overlap_ratio": float(overlap_ratio),
            "center_inside_mask": bool(center_inside),
            "cane_id": cane_id,
            "cane_order": cane_order,
            "cane_distance": cane_distance,
            "patch": extract_patch(
                gray_frame,
                bud_box,
                image_width,
                image_height
            )
        })

    return detections


def prediction_radius(track, predicted_box, regional_reliability):
    """
    Build the normal dynamic matching radius around the position predicted
    entirely from regional cane motion.
    """
    width = max(1.0, predicted_box[2] - predicted_box[0])
    height = max(1.0, predicted_box[3] - predicted_box[1])
    diagonal = math.hypot(width, height)

    radius = max(
        PREDICTION_MIN_RADIUS,
        PREDICTION_BOX_DIAG_MULT * diagonal
    )

    if regional_reliability < 0.25:
        radius += PREDICTION_REGION_WEAK_EXTRA

    radius += track["current_misses"] * PREDICTION_GAP_EXTRA_PER_FRAME

    return min(
        PREDICTION_MAX_RADIUS,
        max(PREDICTION_MIN_RADIUS, radius)
    )


def predict_confirmed_tracks(
    confirmed_tracks,
    regional_motion,
    cane_structures,
    current_frame_idx,
    image_width,
    image_height,
):
    """
    Predict every confirmed bud using 100% regional cane optical flow.

    For each bud:
      1. sample the regional 8x6 cane-motion map near the bud;
      2. apply that movement directly to the bud centre and box;
      3. build the existing dynamic matching radius around that prediction.

    If no reliable regional vector exists, retain the existing velocity fallback.
    """
    predictions = {}

    for stable_id, track in confirmed_tracks.items():
        if track['status'] in {'retired', 'suppressed'}:
            continue

        gap_since_detection = (
            current_frame_idx - track['last_detected_frame']
        )

        if gap_since_detection > LOST_MAX_GAP_FRAMES:
            track['status'] = 'retired'
            continue

        base_center = np.asarray(
            track['state_center'],
            dtype=np.float32
        )
        base_box = track['state_box']

        regional_vector, regional_reliability = sample_regional_motion(
            regional_motion,
            base_center
        )

        if regional_reliability <= 0:
            regional_vector = np.asarray(
                track.get('velocity', (0.0, 0.0)),
                dtype=np.float32
            )

        # 100% regional cane movement. No local OF refinement.
        final_vector = regional_vector.copy()

        predicted_center = base_center + final_vector
        predicted_center = np.asarray(
            clip_point(
                predicted_center,
                image_width,
                image_height
            ),
            dtype=np.float32
        )

        predicted_box = shift_box(
            base_box,
            float(final_vector[0]),
            float(final_vector[1]),
            image_width,
            image_height
        )

        radius = prediction_radius(
            track,
            predicted_box,
            regional_reliability
        )

        (
            predicted_cane_id,
            predicted_cane_order,
            predicted_cane_distance
        ) = assign_point_to_cane(
            predicted_center,
            cane_structures
        )

        predictions[stable_id] = {
            'stable_id': stable_id,
            'center': tuple(map(float, predicted_center)),
            'box': predicted_box,
            'radius': float(radius),
            'regional_vector': tuple(map(float, regional_vector)),
            'regional_reliability': float(regional_reliability),
            'cane_id': predicted_cane_id,
            'cane_order': predicted_cane_order,
            'cane_distance': predicted_cane_distance,
        }

    return predictions


def cane_match_penalty(prediction, detection):
    pred_cane = prediction.get("cane_id")
    det_cane = detection.get("cane_id")

    if pred_cane is None and det_cane is None:
        return 0.20
    if pred_cane is None or det_cane is None:
        return 0.30
    if pred_cane == det_cane:
        return 0.0
    return 1.0


def order_match_penalty(prediction, detection):
    pred_order = prediction.get("cane_order")
    det_order = detection.get("cane_order")

    if pred_order is None or det_order is None:
        return 0.35

    return min(1.0, abs(float(pred_order) - float(det_order)))


def match_confirmed_tracks(
    confirmed_tracks,
    predictions,
    detections,
):
    """
    Match confirmed tracks to current detections.

    """

    track_ids = list(predictions.keys())

    if not track_ids or not detections:
        return [], set(track_ids), set(range(len(detections)))

    cost_matrix = np.full(
        (len(track_ids), len(detections)),
        MATCH_LARGE_COST,
        dtype=np.float32
    )

    for row, stable_id in enumerate(track_ids):
        prediction = predictions[stable_id]
        track = confirmed_tracks[stable_id]
        predicted_center = np.asarray(
            prediction["center"],
            dtype=np.float32
        )

        for col, detection in enumerate(detections):
            # New detections may be off-mask only when reconnecting an old
            # confirmed track. The option can be disabled above.
            if (
                not detection["cane_pass"]
                and not ALLOW_CONFIRMED_RECONNECT_OFF_MASK
            ):
                continue

            detection_center = np.asarray(
                detection["center"],
                dtype=np.float32
            )
            distance = float(
                np.linalg.norm(detection_center - predicted_center)
            )

            if distance > prediction["radius"]:
                continue

            area_ratio = box_area_ratio(
                prediction["box"],
                detection["box"]
            )
            if area_ratio > MATCH_MAX_AREA_RATIO:
                continue

            distance_cost = (
                distance / max(prediction["radius"], 1e-6)
            )
            area_cost = min(
                1.0,
                abs(math.log(area_ratio))
                / math.log(MATCH_MAX_AREA_RATIO)
            )
            iou_cost = (
                1.0
                - box_iou(prediction["box"], detection["box"])
            )
            cane_cost = cane_match_penalty(prediction, detection)
            order_cost = order_match_penalty(prediction, detection)

            similarity = patch_similarity(
                track.get("patch"),
                detection.get("patch")
            )
            patch_cost = 0.5 * (1.0 - similarity)

            total_cost = (
                MATCH_WEIGHT_DISTANCE * distance_cost
                + MATCH_WEIGHT_AREA * area_cost
                + MATCH_WEIGHT_IOU * iou_cost
                + MATCH_WEIGHT_CANE * cane_cost
                + MATCH_WEIGHT_ORDER * order_cost
                + MATCH_WEIGHT_PATCH * patch_cost
            )

            cost_matrix[row, col] = float(total_cost)

    row_indexes, col_indexes = linear_sum_assignment(cost_matrix)
    matches = []
    matched_tracks = set()
    matched_detections = set()

    for row, col in zip(row_indexes, col_indexes):
        cost = float(cost_matrix[row, col])
        if cost >= MATCH_LARGE_COST or cost > MATCH_MAX_COST:
            continue

        stable_id = track_ids[row]
        matches.append({
            "stable_id": stable_id,
            "detection_index": int(col),
            "cost": cost
        })
        matched_tracks.add(stable_id)
        matched_detections.add(int(col))

    unmatched_tracks = set(track_ids) - matched_tracks
    unmatched_detections = (
        set(range(len(detections))) - matched_detections
    )

    return matches, unmatched_tracks, unmatched_detections


def create_pending_track(temp_id, detection, frame_idx):
    return {
        "temp_id": temp_id,
        "first_frame": frame_idx,
        "last_seen_frame": frame_idx,
        "state_center": detection["center"],
        "state_box": detection["box"].copy(),
        "velocity": (0.0, 0.0),
        "hit_frames": deque([frame_idx], maxlen=PENDING_CONFIRM_WINDOW),
        "observation_count": 1,
        "clear_observations": 1,
        "current_misses": 0,
        "best_conf": detection["conf"],
        "best_box": detection["box"].copy(),
        "best_overlap": detection["mask_overlap_ratio"],
        "best_center_inside": detection["center_inside_mask"],
        "cane_id": detection["cane_id"],
        "cane_order": detection["cane_order"],
        "patch": detection["patch"],
        "last_detection_index": None
    }


def predict_pending_tracks(pending_tracks, regional_motion, cane_structures):
    predictions = {}
    for temp_id, pending in pending_tracks.items():
        vector, reliability = sample_regional_motion(regional_motion, pending['state_center'])
        if reliability <= 0:
            vector = np.asarray(pending.get('velocity', (0.0, 0.0)), dtype=np.float32)
        center = np.asarray(pending['state_center'], dtype=np.float32) + vector
        center = np.asarray(clip_point(center, W, H), dtype=np.float32)
        box = shift_box(pending['state_box'], float(vector[0]), float(vector[1]), W, H)
        cane_id, cane_order, cane_distance = assign_point_to_cane(center, cane_structures)
        predictions[temp_id] = {'center': tuple(map(float, center)), 'box': box, 'radius': PENDING_MATCH_RADIUS + 10.0 * pending['current_misses'], 'cane_id': cane_id, 'cane_order': cane_order, 'cane_distance': cane_distance, 'regional_reliability': float(reliability)}
    return predictions


def update_pending_tracks(
    pending_tracks,
    unmatched_detection_indexes,
    detections,
    regional_motion,
    cane_structures,
    current_frame_idx,
    next_pending_id,
):
    """
    Update pending tracks and apply the existing 2-of-3 confirmation rule.

    """

    pending_predictions = predict_pending_tracks(
        pending_tracks,
        regional_motion,
        cane_structures
    )
    temp_ids = list(pending_predictions.keys())
    detection_indexes = [
        index
        for index in sorted(unmatched_detection_indexes)
        if detections[index]['cane_pass']
    ]
    matches = []

    if temp_ids and detection_indexes:
        cost_matrix = np.full(
            (len(temp_ids), len(detection_indexes)),
            MATCH_LARGE_COST,
            dtype=np.float32
        )

        for row, temp_id in enumerate(temp_ids):
            prediction = pending_predictions[temp_id]
            pending = pending_tracks[temp_id]
            pred_center = np.asarray(
                prediction['center'],
                dtype=np.float32
            )

            for col, detection_index in enumerate(detection_indexes):
                detection = detections[detection_index]
                det_center = np.asarray(
                    detection['center'],
                    dtype=np.float32
                )
                distance = float(
                    np.linalg.norm(det_center - pred_center)
                )

                if distance > prediction['radius']:
                    continue

                area_ratio = box_area_ratio(
                    prediction['box'],
                    detection['box']
                )
                if area_ratio > PENDING_MATCH_MAX_AREA_RATIO:
                    continue

                distance_cost = (
                    distance / max(prediction['radius'], 1e-06)
                )
                area_cost = min(
                    1.0,
                    abs(math.log(area_ratio))
                    / math.log(PENDING_MATCH_MAX_AREA_RATIO)
                )
                cane_cost = cane_match_penalty(
                    prediction,
                    detection
                )
                similarity = patch_similarity(
                    pending.get('patch'),
                    detection.get('patch')
                )
                patch_cost = 0.5 * (1.0 - similarity)

                total_cost = (
                    0.65 * distance_cost
                    + 0.15 * area_cost
                    + 0.12 * cane_cost
                    + 0.08 * patch_cost
                )
                cost_matrix[row, col] = total_cost

        rows, cols = linear_sum_assignment(cost_matrix)
        for row, col in zip(rows, cols):
            cost = float(cost_matrix[row, col])
            if (
                cost >= MATCH_LARGE_COST
                or cost > PENDING_MATCH_MAX_COST
            ):
                continue
            matches.append(
                (temp_ids[row], detection_indexes[col], cost)
            )

    matched_pending = set()
    matched_detection_indexes = set()

    for temp_id, detection_index, cost in matches:
        pending = pending_tracks[temp_id]
        detection = detections[detection_index]
        previous_center = np.asarray(
            pending['state_center'],
            dtype=np.float32
        )
        current_center = np.asarray(
            detection['center'],
            dtype=np.float32
        )
        velocity = current_center - previous_center

        pending['state_center'] = detection['center']
        pending['state_box'] = detection['box'].copy()
        pending['velocity'] = tuple(map(float, velocity))
        pending['last_seen_frame'] = current_frame_idx
        pending['hit_frames'].append(current_frame_idx)
        pending['observation_count'] += 1
        pending['clear_observations'] += 1
        pending['current_misses'] = 0
        pending['cane_id'] = detection['cane_id']
        pending['cane_order'] = detection['cane_order']
        pending['patch'] = detection['patch']
        pending['last_detection_index'] = detection_index

        if detection['conf'] > pending['best_conf']:
            pending['best_conf'] = detection['conf']
            pending['best_box'] = detection['box'].copy()
            pending['best_overlap'] = (
                detection['mask_overlap_ratio']
            )
            pending['best_center_inside'] = (
                detection['center_inside_mask']
            )

        matched_pending.add(temp_id)
        matched_detection_indexes.add(detection_index)

    expired = []
    for temp_id, pending in pending_tracks.items():
        if temp_id in matched_pending:
            continue

        prediction = pending_predictions.get(temp_id)
        if prediction is not None:
            pending['state_center'] = prediction['center']
            pending['state_box'] = prediction['box']

        pending['current_misses'] += 1
        pending['last_detection_index'] = None
        age = current_frame_idx - pending['first_frame']

        if (
            age > PENDING_MAX_AGE_FRAMES
            or pending['current_misses'] > PENDING_MAX_MISSES
        ):
            expired.append(temp_id)

    for temp_id in expired:
        pending_tracks.pop(temp_id, None)

    remaining_detection_indexes = [
        index
        for index in detection_indexes
        if index not in matched_detection_indexes
    ]

    for detection_index in remaining_detection_indexes:
        detection = detections[detection_index]
        temp_id = next_pending_id
        next_pending_id += 1
        pending_tracks[temp_id] = create_pending_track(
            temp_id,
            detection,
            current_frame_idx
        )
        pending_tracks[temp_id][
            'last_detection_index'
        ] = detection_index
        matched_detection_indexes.add(detection_index)

    confirmations = []

    for temp_id, pending in list(pending_tracks.items()):
        recent_hits = [
            frame
            for frame in pending['hit_frames']
            if current_frame_idx - frame < PENDING_CONFIRM_WINDOW
        ]

        if len(set(recent_hits)) < PENDING_CONFIRM_HITS:
            continue
        if pending['last_seen_frame'] != current_frame_idx:
            continue

        confirmations.append(pending.copy())
        pending_tracks.pop(temp_id, None)

    return pending_tracks, confirmations, next_pending_id


def create_confirmed_track(stable_id, detection, frame_idx, pending_summary=None):
    if pending_summary is None:
        first_frame = frame_idx
        frames_seen = 1
        best_conf = detection['conf']
        best_box = detection['box'].copy()
        best_overlap = detection['mask_overlap_ratio']
        best_center_inside = detection['center_inside_mask']
    else:
        first_frame = int(pending_summary['first_frame'])
        frames_seen = int(pending_summary['observation_count'])
        best_conf = float(pending_summary['best_conf'])
        best_box = pending_summary['best_box'].copy()
        best_overlap = float(pending_summary['best_overlap'])
        best_center_inside = bool(pending_summary['best_center_inside'])
    clear_observations = (
        int(pending_summary.get('clear_observations', frames_seen))
        if pending_summary is not None
        else 1
    )
    return {
        'stable_bud_id': stable_id,
        'status': 'active',
        'first_frame': first_frame,
        'last_detected_frame': frame_idx,
        'last_state_frame': frame_idx,
        'state_center': detection['center'],
        'state_box': detection['box'].copy(),
        'last_detected_box': detection['box'].copy(),
        'velocity': (0.0, 0.0),
        'current_misses': 0,
        'total_missed_frames': 0,
        'max_consecutive_misses': 0,
        'frames_seen': frames_seen,
        'clear_observations': clear_observations,
        'counted': True,
        'counted_frame': frame_idx,
        'best_conf': best_conf,
        'best_box': best_box,
        'best_mask_overlap_ratio': best_overlap,
        'best_center_inside_mask': best_center_inside,
        'cane_id': detection['cane_id'],
        'cane_order': detection['cane_order'],
        'patch': detection['patch'],
        'regional_predictions': 0,
        'global_matches': 0,
        'off_mask_reconnections': 0,
        'first_match_cost': None,
        'last_match_cost': None
    }


def update_matched_confirmed_track(
    track,
    detection,
    prediction,
    match_cost,
    current_frame_idx,
):
    previous_center = np.asarray(
        track['state_center'],
        dtype=np.float32
    )
    current_center = np.asarray(
        detection['center'],
        dtype=np.float32
    )
    velocity = current_center - previous_center

    track['state_center'] = detection['center']
    track['state_box'] = detection['box'].copy()
    track['last_detected_box'] = detection['box'].copy()
    track['velocity'] = tuple(map(float, velocity))
    track['last_detected_frame'] = current_frame_idx
    track['last_state_frame'] = current_frame_idx
    track['current_misses'] = 0
    track['frames_seen'] += 1
    track['clear_observations'] = (
        track.get('clear_observations', 0) + 1
    )
    track['status'] = 'active'
    track['cane_id'] = detection['cane_id']
    track['cane_order'] = detection['cane_order']
    track['patch'] = detection['patch']
    track['global_matches'] += 1
    track['last_match_cost'] = float(match_cost)

    if track['first_match_cost'] is None:
        track['first_match_cost'] = float(match_cost)

    if prediction is not None:
        track['regional_predictions'] += 1

    if not detection['cane_pass']:
        track['off_mask_reconnections'] += 1

    if detection['conf'] > track['best_conf']:
        track['best_conf'] = detection['conf']
        track['best_box'] = detection['box'].copy()
        track['best_mask_overlap_ratio'] = (
            detection['mask_overlap_ratio']
        )
        track['best_center_inside_mask'] = (
            detection['center_inside_mask']
        )


def update_unmatched_confirmed_track(
    track,
    prediction,
    current_frame_idx,
):
    if prediction is None:
        track['status'] = 'retired'
        return

    track['state_center'] = prediction['center']
    track['state_box'] = prediction['box']
    track['last_state_frame'] = current_frame_idx
    track['current_misses'] += 1
    track['total_missed_frames'] += 1
    track['max_consecutive_misses'] = max(
        track['max_consecutive_misses'],
        track['current_misses']
    )
    track['status'] = 'lost'
    track['regional_predictions'] += 1

    if track['current_misses'] > LOST_MAX_GAP_FRAMES:
        track['status'] = 'retired'


def make_countable_first_year_mask(first_year_mask, male_mask):
    """
    Male has priority. Any pixel claimed as male is removed from the
    first-year mask before bud acceptance and regional cane optical flow.
    """
    first_year_binary = (first_year_mask > 0).astype(np.uint8) * 255
    male_binary = (male_mask > 0).astype(np.uint8) * 255
    return cv2.bitwise_and(first_year_binary, cv2.bitwise_not(male_binary))

# ============================================================
# 7) PROCESSED-FRAME AND OPTIONAL-VIDEO DRAWING
# ============================================================


def draw_box_only(image, box, color, thickness=BUD_BOX_THICKNESS):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_roi_lines(image, include_roi, counting_enabled=True):
    """
    Draw only the active counting-ROI rectangle on processed JPGs.

    No text label and no grey masking are added. If the full frame is
    countable (include_roi is None), no rectangle is needed. If counting is
    disabled for the frame, no ROI line is drawn.
    """
    if not counting_enabled or include_roi is None:
        return

    x1, y1, x2, y2 = map(int, include_roi)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        ROI_LINE_COLOR,
        ROI_LINE_THICKNESS,
    )


def apply_colour_coded_masks(frame, male_mask, first_year_mask):
    """
    Create diagnostic cane highlighting matching the test pipeline.

    Male cane       = light orange
    First-year wood = light green
    Male cane takes priority anywhere the masks overlap.
    """
    output = frame.copy()

    male_pixels = male_mask > 0
    first_year_pixels = first_year_mask > 0
    green_pixels = first_year_pixels & ~male_pixels

    if np.any(green_pixels):
        original_pixels = frame[green_pixels].astype(np.float32)
        colour = np.asarray(FIRST_YEAR_COLOR, dtype=np.float32)
        blended = (
            (1.0 - MASK_ALPHA) * original_pixels
            + MASK_ALPHA * colour
        )
        output[green_pixels] = np.clip(
            blended,
            0,
            255,
        ).astype(np.uint8)

    if np.any(male_pixels):
        original_pixels = frame[male_pixels].astype(np.float32)
        colour = np.asarray(MALE_COLOR, dtype=np.float32)
        blended = (
            (1.0 - MASK_ALPHA) * original_pixels
            + MASK_ALPHA * colour
        )
        output[male_pixels] = np.clip(
            blended,
            0,
            255,
        ).astype(np.uint8)

    return output


def draw_labelled_box(image, box, label, color, thickness=BUD_BOX_THICKNESS):
    """Draw one labelled bud box for the optional diagnostic videos."""
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    if not label:
        return

    origin = (x1, max(18, y1 - 6))

    # Black outline improves text readability against the orchard background.
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_frame_summary(image, bud_visible, bud_total, frame_id):
    """Draw only the retained winter-bud count and frame-ID summary."""
    if not DRAW_FRAME_SUMMARY:
        return

    lines = [
        f"Winter buds visible: {int(bud_visible)}",
        f"Total unique winter buds: {int(bud_total)}",
        f"Frame ID: {int(frame_id)}",
    ]

    font = cv2.FONT_HERSHEY_SIMPLEX
    text_sizes = [
        cv2.getTextSize(
            line,
            font,
            FRAME_SUMMARY_FONT_SCALE,
            FRAME_SUMMARY_FONT_THICKNESS,
        )[0]
        for line in lines
    ]

    max_text_width = max(width for width, _ in text_sizes)
    line_height = max(height for _, height in text_sizes)
    box_width = max_text_width + 2 * FRAME_SUMMARY_PADDING
    box_height = (
        2 * FRAME_SUMMARY_PADDING
        + len(lines) * line_height
        + (len(lines) - 1) * FRAME_SUMMARY_LINE_GAP
    )

    image_height, image_width = image.shape[:2]
    x1 = min(max(0, FRAME_SUMMARY_MARGIN), max(0, image_width - 1))
    y1 = min(max(0, FRAME_SUMMARY_MARGIN), max(0, image_height - 1))
    x2 = min(image_width, x1 + box_width)
    y2 = min(image_height, y1 + box_height)

    if x2 <= x1 or y2 <= y1:
        return

    roi = image[y1:y2, x1:x2]
    background = np.empty_like(roi)
    background[:] = FRAME_SUMMARY_BACKGROUND_COLOR
    cv2.addWeighted(
        background,
        FRAME_SUMMARY_BACKGROUND_ALPHA,
        roi,
        1.0 - FRAME_SUMMARY_BACKGROUND_ALPHA,
        0.0,
        dst=roi,
    )

    text_x = x1 + FRAME_SUMMARY_PADDING
    baseline_y = y1 + FRAME_SUMMARY_PADDING + line_height

    for line_index, line in enumerate(lines):
        text_y = baseline_y + line_index * (
            line_height + FRAME_SUMMARY_LINE_GAP
        )
        cv2.putText(
            image,
            line,
            (text_x, text_y),
            font,
            FRAME_SUMMARY_FONT_SCALE,
            FRAME_SUMMARY_TEXT_COLOR,
            FRAME_SUMMARY_FONT_THICKNESS,
            cv2.LINE_AA,
        )


def draw_current_bud_boxes(
    image,
    visible_confirmed_buds,
    pending_tracks,
    detections,
    current_frame_idx,
):
    """Draw boxes only: confirmed buds plus current pending candidates."""
    for item in visible_confirmed_buds:
        detection = item["detection"]
        color = (
            BUD_NEW_COLOR
            if item.get("newly_confirmed", False)
            else BUD_OLD_COLOR
        )
        draw_box_only(
            image,
            detection["box"],
            color,
        )

    for pending in pending_tracks.values():
        if pending.get("last_seen_frame") != current_frame_idx:
            continue

        detection_index = pending.get("last_detection_index")

        if detection_index is None:
            continue

        detection_index = int(detection_index)

        if not (0 <= detection_index < len(detections)):
            continue

        draw_box_only(
            image,
            detections[detection_index]["box"],
            BUD_WAITING_COLOR,
        )


def draw_current_buds_with_ids(
    image,
    visible_confirmed_buds,
    pending_tracks,
    detections,
    current_frame_idx,
):
    """
    Draw the same labelled winter-bud overlay used by the test-code videos.

    Confirmed stable buds:
      B<stable_id>

    Current pending/unconfirmed buds:
      P<pending_id>
    """
    for item in visible_confirmed_buds:
        stable_id = int(item["stable_id"])
        detection = item["detection"]
        is_new = bool(item.get("newly_confirmed", False))

        color = BUD_NEW_COLOR if is_new else BUD_OLD_COLOR

        draw_labelled_box(
            image,
            detection["box"],
            f"B{stable_id}",
            color,
            BUD_BOX_THICKNESS,
        )

    for temp_id, pending in pending_tracks.items():
        if pending.get("last_seen_frame") != current_frame_idx:
            continue

        detection_index = pending.get("last_detection_index")

        if detection_index is None:
            continue

        detection_index = int(detection_index)

        if not (0 <= detection_index < len(detections)):
            continue

        draw_labelled_box(
            image,
            detections[detection_index]["box"],
            f"P{temp_id}",
            BUD_WAITING_COLOR,
            BUD_BOX_THICKNESS,
        )


# ============================================================
# 8) INPUT VALIDATION, DEVICE, MODELS AND VIDEO
# ============================================================

for required_model in (MALE_MODEL_PATH, CANE_MODEL_PATH, BUD_MODEL_PATH):
    if not os.path.isfile(required_model):
        raise FileNotFoundError(f"Model not found: {required_model}")

if not os.path.isfile(VIDEO_PATH):
    raise FileNotFoundError(f"Input video not found: {VIDEO_PATH}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU was not detected. The Cutie Step 3 pipeline requires a "
        "CUDA-enabled SageMaker instance/environment."
    )

DEVICE = 0
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

print("==========================================")
print("GPU inference enabled")
print("GPU:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
print("==========================================")

print("\nLoading YOLO models...")
male_model = YOLO(MALE_MODEL_PATH)
cane_model = YOLO(CANE_MODEL_PATH)
bud_model = YOLO(BUD_MODEL_PATH)

FIRST_YEAR_CLASS_ID = get_class_id_by_name(
    cane_model,
    FIRST_YEAR_CLASS_NAME,
)

if MALE_CLASS_ID not in male_model.names:
    raise ValueError(
        f"MALE_CLASS_ID={MALE_CLASS_ID} was not found in male model classes: "
        f"{male_model.names}"
    )
if BUD_CLASS_ID not in bud_model.names:
    raise ValueError(
        f"BUD_CLASS_ID={BUD_CLASS_ID} was not found in bud model classes: "
        f"{bud_model.names}"
    )

print("YOLO models loaded successfully.")
print("Male model classes:", male_model.names)
print("Cane model classes:", cane_model.names)
print("Bud model classes:", bud_model.names)
print("Using male cane class:", MALE_CLASS_ID, male_model.names[MALE_CLASS_ID])
print(
    "Using first-year wood class:",
    FIRST_YEAR_CLASS_ID,
    cane_model.names[FIRST_YEAR_CLASS_ID],
)
print("Using winter-bud class:", BUD_CLASS_ID, bud_model.names[BUD_CLASS_ID])

print("\nLoading Cutie model...")
from hydra.core.global_hydra import GlobalHydra
if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()
cutie_model = get_default_model()
print("Cutie model loaded successfully.")
print("Cutie repository:", CUTIE_REPO)
print("Cutie internal shorter-edge size:", CUTIE_MAX_INTERNAL_SIZE)
print("Cutie correction interval:", CUTIE_CORRECTION_INTERVAL, "frames")

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps is None or fps <= 0:
    fps = 30.0

source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames <= 0:
    total_frames = None

W = LANDSCAPE_FRAME_WIDTH
H = LANDSCAPE_FRAME_HEIGHT

if MAX_FRAMES_TO_PROCESS is None:
    process_frames = total_frames
else:
    process_frames = (
        min(total_frames, MAX_FRAMES_TO_PROCESS)
        if total_frames is not None
        else MAX_FRAMES_TO_PROCESS
    )

DEFAULT_INCLUDE_ROI = parse_roi_coordinates(
    INCLUDE_ROI_TEXT,
    W,
    H,
    argument_label="--include_roi",
)
INCLUDE_ROI_RANGES = parse_include_roi_ranges(
    INCLUDE_ROI_RANGES_TEXT,
    W,
    H,
    video_frame_count=total_frames,
)


def get_active_counting_roi(frame_id):
    frame_id = int(frame_id)
    for range_item in INCLUDE_ROI_RANGES:
        if range_item["start_frame"] <= frame_id <= range_item["end_frame"]:
            return (
                True,
                range_item["roi"],
                (range_item["start_frame"], range_item["end_frame"]),
            )

    if DEFAULT_INCLUDE_ROI is not None:
        return True, DEFAULT_INCLUDE_ROI, None
    if INCLUDE_ROI_RANGES:
        return False, None, None
    return True, None, None


if INCLUDE_ROI_RANGES:
    print("Frame-specific counting ROI ranges (landscape 3840 x 2160):")
    for range_item in INCLUDE_ROI_RANGES:
        print(
            f"  Frames {range_item['start_frame']}-{range_item['end_frame']}: "
            + ",".join(map(str, range_item["roi"]))
        )
    if DEFAULT_INCLUDE_ROI is None:
        print("Frames outside the listed ranges: counting disabled.")
    else:
        print(
            "Fallback ROI for frames outside the listed ranges: ",
            ",".join(map(str, DEFAULT_INCLUDE_ROI)),
        )
elif DEFAULT_INCLUDE_ROI is None:
    print("Counting ROI: complete landscape 3840 x 2160 frame")
else:
    print(
        "Counting ROI for the complete video: ",
        ",".join(map(str, DEFAULT_INCLUDE_ROI)),
    )

# Derive reduced corridor and regional-LK dimensions.
CANE_CORRIDOR_W = max(1, int(round(W * CANE_CORRIDOR_RESOLUTION_SCALE)))
CANE_CORRIDOR_H = max(1, int(round(H * CANE_CORRIDOR_RESOLUTION_SCALE)))
CANE_CORRIDOR_SCALE_X = CANE_CORRIDOR_W / float(W)
CANE_CORRIDOR_SCALE_Y = CANE_CORRIDOR_H / float(H)
CANE_CORRIDOR_RADIUS_X_SCALED = max(
    1,
    int(math.ceil(CANE_CORRIDOR_RADIUS * CANE_CORRIDOR_SCALE_X)),
)
CANE_CORRIDOR_RADIUS_Y_SCALED = max(
    1,
    int(math.ceil(CANE_CORRIDOR_RADIUS * CANE_CORRIDOR_SCALE_Y)),
)

REGION_FLOW_W = max(1, int(round(W * REGION_FLOW_RESOLUTION_SCALE)))
REGION_FLOW_H = max(1, int(round(H * REGION_FLOW_RESOLUTION_SCALE)))
REGION_FLOW_SCALE_X = REGION_FLOW_W / float(W)
REGION_FLOW_SCALE_Y = REGION_FLOW_H / float(H)
REGION_FLOW_MIN_SCALE = min(REGION_FLOW_SCALE_X, REGION_FLOW_SCALE_Y)
REGION_FLOW_MAX_SCALE = max(REGION_FLOW_SCALE_X, REGION_FLOW_SCALE_Y)
REGION_LK_WIN_SIZE_SCALED = scaled_odd_size(
    REGION_LK_WIN_SIZE,
    REGION_FLOW_MIN_SCALE,
    minimum=3,
)
REGION_BLOCK_SIZE_SCALED = scaled_odd_size(
    REGION_BLOCK_SIZE,
    REGION_FLOW_MIN_SCALE,
    minimum=3,
)
REGION_MIN_DISTANCE_SCALED = max(
    1.0,
    REGION_MIN_DISTANCE * REGION_FLOW_MIN_SCALE,
)
REGION_FB_MAX_ERROR_SCALED = max(
    0.25,
    REGION_FB_MAX_ERROR * REGION_FLOW_MAX_SCALE,
)
REGION_MAX_REASONABLE_DISPLACEMENT_SCALED = max(
    1.0,
    REGION_MAX_REASONABLE_DISPLACEMENT * REGION_FLOW_MAX_SCALE,
)

print("\nVideo loaded successfully.")
print("Source frame size reported by OpenCV:", source_width, "x", source_height)
print("Working frame size after orientation normalisation:", W, "x", H)
print("Portrait handling: 2160 x 3840 -> rotate 90 degrees clockwise")
print("Landscape handling: 3840 x 2160 -> unchanged")
print("FPS:", fps)
print("Frames to process:", process_frames if process_frames else "until end")
print("Male tracker: Cutie")
print("First-year tracker: Cutie")
print("Regional optical-flow size:", f"{REGION_FLOW_W} x {REGION_FLOW_H}")
print("Bud confirmation:", f"{PENDING_CONFIRM_HITS} of {PENDING_CONFIRM_WINDOW} frames")
print("Bud lost-track memory:", LOST_MAX_GAP_FRAMES, "frames")
print("Local per-bud optical flow: disabled")
print("Spur detection/tracking: disabled")
print("Optional diagnostic videos:", "enabled" if SAVE_OUTPUT_VIDEOS else "disabled")


# ------------------------------------------------------------
# OPTIONAL TEST-STYLE OUTPUT VIDEOS
# ------------------------------------------------------------

diagnostic_writer = None
bud_only_writer = None

if SAVE_OUTPUT_VIDEOS:
    for output_path in (OUTPUT_VIDEO, OUTPUT_BUD_ONLY_VIDEO):
        output_parent = os.path.dirname(output_path)

        if output_parent:
            os.makedirs(output_parent, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    diagnostic_writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        fourcc,
        fps,
        (W, H),
    )

    bud_only_writer = cv2.VideoWriter(
        OUTPUT_BUD_ONLY_VIDEO,
        fourcc,
        fps,
        (W, H),
    )

    if (
        not diagnostic_writer.isOpened()
        or not bud_only_writer.isOpened()
    ):
        if diagnostic_writer is not None:
            diagnostic_writer.release()

        if bud_only_writer is not None:
            bud_only_writer.release()

        cap.release()

        raise RuntimeError(
            "Could not create one or both optional Step 3 output videos. "
            "Check that OpenCV has MP4 encoding support and that the output "
            "paths are writable."
        )

    print("Diagnostic cane-highlighted video:", OUTPUT_VIDEO)
    print("Diagnostic bud-only video:", OUTPUT_BUD_ONLY_VIDEO)


# ============================================================
# 9) TRACKING STATE
# ============================================================

confirmed_tracks = {}
pending_tracks = {}
next_stable_id = 1
next_pending_id = 1

male_cutie_state = make_cutie_stream_state("male cane")
first_year_cutie_state = make_cutie_stream_state("first-year wood")

previous_regional_gray = None
previous_regional_corridor_mask = None

frame_rows = []
timing_rows = []
male_detection_instances = 0
first_year_detection_instances = 0
bud_detection_instances = 0


# ============================================================
# 10) MAIN LOOP
# ============================================================

start_time = time.time()
frame_idx = 0

pbar = tqdm(
    total=process_frames,
    desc="Processing winter buds",
    unit="frame",
    dynamic_ncols=True,
)

try:
    while True:
        if process_frames is not None and frame_idx >= process_frames:
            break

        frame_total_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        frame = normalise_frame_orientation(frame)
        (
            counting_enabled,
            active_include_roi,
            active_roi_frame_range,
        ) = get_active_counting_roi(frame_idx)

        current_gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ----------------------------------------------------
        # A) MALE-CANE YOLO
        # ----------------------------------------------------
        profile_cuda_sync()
        stage_start = time.perf_counter()
        with torch.inference_mode():
            male_results = male_model.predict(
                source=frame,
                task="segment",
                conf=MALE_CONF_THRES,
                iou=MALE_IOU_THRES,
                imgsz=MALE_IMG_SIZE,
                classes=[MALE_CLASS_ID],
                max_det=MALE_MAX_DET,
                device=DEVICE,
                verbose=False,
            )
        profile_cuda_sync()
        male_inference_ms = elapsed_ms(stage_start)

        yolo_male_mask = np.zeros((H, W), dtype=np.uint8)
        if male_results:
            male_result = male_results[0]
            if male_result.boxes is not None and len(male_result.boxes) > 0:
                male_detection_instances += len(male_result.boxes)
            add_polygons_to_mask(
                male_result,
                yolo_male_mask,
                min_area=0,
            )

        # ----------------------------------------------------
        # B) FIRST-YEAR-CANE YOLO
        # ----------------------------------------------------
        profile_cuda_sync()
        stage_start = time.perf_counter()
        with torch.inference_mode():
            cane_results = cane_model.predict(
                source=frame,
                task="segment",
                conf=CANE_CONF_THRES,
                iou=CANE_IOU_THRES,
                imgsz=CANE_IMG_SIZE,
                classes=[FIRST_YEAR_CLASS_ID],
                max_det=CANE_MAX_DET,
                device=DEVICE,
                retina_masks=True,
                verbose=False,
            )
        profile_cuda_sync()
        cane_inference_ms = elapsed_ms(stage_start)

        yolo_first_year_mask = np.zeros((H, W), dtype=np.uint8)
        if cane_results:
            cane_result = cane_results[0]
            if cane_result.boxes is not None and len(cane_result.boxes) > 0:
                first_year_detection_instances += len(cane_result.boxes)
            add_polygons_to_mask(
                cane_result,
                yolo_first_year_mask,
                min_area=CANE_MIN_MASK_AREA,
            )

        yolo_first_year_mask = clean_yolo_first_year_mask(
            yolo_first_year_mask
        )

        # ----------------------------------------------------
        # C) TWO INDEPENDENT CUTIE CANE STREAMS
        # ----------------------------------------------------
        stage_start = time.perf_counter()
        cutie_image = frame_to_cutie_tensor(frame)

        male_mask = update_cutie_stream(
            state=male_cutie_state,
            cutie_model=cutie_model,
            image_tensor=cutie_image,
            yolo_mask=yolo_male_mask,
            frame_idx=frame_idx,
            frame_height=H,
            frame_width=W,
        )
        first_year_mask = update_cutie_stream(
            state=first_year_cutie_state,
            cutie_model=cutie_model,
            image_tensor=cutie_image,
            yolo_mask=yolo_first_year_mask,
            frame_idx=frame_idx,
            frame_height=H,
            frame_width=W,
        )
        profile_cuda_sync()
        cutie_tracking_ms = elapsed_ms(stage_start)

        # ----------------------------------------------------
        # D) COUNTABLE FIRST-YEAR MASK AND MATCHING STRUCTURES
        # ----------------------------------------------------
        stage_start = time.perf_counter()

        countable_first_year_mask = make_countable_first_year_mask(
            first_year_mask,
            male_mask,
        )
        acceptance_mask = create_acceptance_mask(
            countable_first_year_mask
        )
        corridor_mask = create_cane_corridor(
            countable_first_year_mask
        )
        cane_structures = build_cane_structures(
            countable_first_year_mask
        )

        if corridor_mask.shape[:2] == (REGION_FLOW_H, REGION_FLOW_W):
            regional_corridor_mask = corridor_mask
        else:
            regional_corridor_mask = cv2.resize(
                corridor_mask,
                (REGION_FLOW_W, REGION_FLOW_H),
                interpolation=cv2.INTER_NEAREST,
            )

        mask_processing_ms = elapsed_ms(stage_start)

        # ----------------------------------------------------
        # E) REGIONAL LK OPTICAL FLOW ONLY
        # ----------------------------------------------------
        stage_start = time.perf_counter()

        current_regional_gray = cv2.resize(
            current_gray_full,
            (REGION_FLOW_W, REGION_FLOW_H),
            interpolation=cv2.INTER_AREA,
        )

        regional_motion = estimate_regional_motion(
            previous_regional_gray,
            current_regional_gray,
            previous_regional_corridor_mask,
            W,
            H,
        )

        optical_flow_ms = elapsed_ms(stage_start)

        lk_dx_px = float(regional_motion.get("lk_dx_px", 0.0))
        lk_dy_px = float(regional_motion.get("lk_dy_px", 0.0))
        lk_shift_px = float(regional_motion.get("lk_shift_px", 0.0))
        lk_tracks_used = int(regional_motion.get("lk_tracks_used", 0))
        lk_ok = int(bool(regional_motion.get("lk_ok", False)))

        # ----------------------------------------------------
        # F) BUD YOLO + ROI/CANE ACCEPTANCE
        # ----------------------------------------------------
        profile_cuda_sync()
        stage_start = time.perf_counter()
        bud_results = run_bud_inference(frame)
        profile_cuda_sync()
        bud_inference_ms = elapsed_ms(stage_start)

        if (
            bud_results
            and bud_results[0].boxes is not None
            and len(bud_results[0].boxes) > 0
        ):
            bud_detection_instances += len(bud_results[0].boxes)

        bud_detections = postprocess_bud_results(
            bud_results,
            current_gray_full,
            acceptance_mask,
            cane_structures,
            W,
            H,
            include_roi=active_include_roi,
            counting_enabled=counting_enabled,
        )

        bud_raw_count = sum(
            1 for detection in bud_detections if detection["cane_pass"]
        )

        # ----------------------------------------------------
        # G) REGIONAL PREDICTION + HUNGARIAN MATCHING + 2/3
        # ----------------------------------------------------
        stage_start = time.perf_counter()

        predictions = predict_confirmed_tracks(
            confirmed_tracks,
            regional_motion,
            cane_structures,
            frame_idx,
            W,
            H,
        )

        (
            matches,
            unmatched_track_ids,
            unmatched_detection_indexes,
        ) = match_confirmed_tracks(
            confirmed_tracks,
            predictions,
            bud_detections,
        )

        visible_confirmed_buds = []

        for match in matches:
            stable_id = match["stable_id"]
            detection_index = match["detection_index"]
            detection = bud_detections[detection_index]
            prediction = predictions.get(stable_id)

            update_matched_confirmed_track(
                confirmed_tracks[stable_id],
                detection,
                prediction,
                match["cost"],
                frame_idx,
            )

            visible_confirmed_buds.append({
                "stable_id": stable_id,
                "detection": detection,
                "newly_confirmed": False,
            })

        for stable_id in unmatched_track_ids:
            prediction = predictions.get(stable_id)
            update_unmatched_confirmed_track(
                confirmed_tracks[stable_id],
                prediction,
                frame_idx,
            )

        for track in confirmed_tracks.values():
            if track["status"] == "retired":
                continue
            if (
                track["stable_bud_id"] not in predictions
                and frame_idx - track["last_detected_frame"]
                > LOST_MAX_GAP_FRAMES
            ):
                track["status"] = "retired"

        (
            pending_tracks,
            confirmations,
            next_pending_id,
        ) = update_pending_tracks(
            pending_tracks,
            unmatched_detection_indexes,
            bud_detections,
            regional_motion,
            cane_structures,
            frame_idx,
            next_pending_id,
        )

        new_ids = []
        for pending_summary in confirmations:
            detection_index = pending_summary.get("last_detection_index")
            if detection_index is None:
                continue

            detection_index = int(detection_index)
            if not (0 <= detection_index < len(bud_detections)):
                continue

            detection = bud_detections[detection_index]
            stable_id = next_stable_id
            next_stable_id += 1

            confirmed_tracks[stable_id] = create_confirmed_track(
                stable_id,
                detection,
                frame_idx,
                pending_summary=pending_summary,
            )
            confirmed_tracks[stable_id]["counted"] = True
            confirmed_tracks[stable_id]["counted_frame"] = frame_idx

            visible_confirmed_buds.append({
                "stable_id": stable_id,
                "detection": detection,
                "newly_confirmed": True,
            })
            new_ids.append(stable_id)

        bud_new_count = len(new_ids)
        bud_old_count = sum(
            1
            for item in visible_confirmed_buds
            if not item.get("newly_confirmed", False)
        )
        bud_total_unique_count = len(confirmed_tracks)

        matching_ms = elapsed_ms(stage_start)

        # ----------------------------------------------------
        # H) DRAW PROCESSED JPG + OPTIONAL TEST-STYLE VIDEOS
        # ----------------------------------------------------
        stage_start = time.perf_counter()

        # ----------------------------------------------------
        # Processed JPG used downstream by Step 6:
        #   - original frame
        #   - bud boxes only
        #   - active ROI rectangle
        #   - winter-bud/frame-ID summary
        #   - no bud ID/confidence text
        #   - no cane-mask highlighting
        # ----------------------------------------------------
        annotated = frame.copy()

        draw_current_bud_boxes(
            annotated,
            visible_confirmed_buds,
            pending_tracks,
            bud_detections,
            frame_idx,
        )

        draw_roi_lines(
            annotated,
            active_include_roi,
            counting_enabled=counting_enabled,
        )

        draw_frame_summary(
            annotated,
            bud_visible=len(visible_confirmed_buds),
            bud_total=bud_total_unique_count,
            frame_id=frame_idx,
        )

        if SAVE_PROCESSED_FRAMES:
            cv2.imwrite(
                os.path.join(
                    PROCESSED_FRAMES_DIR,
                    f"frame_{frame_idx:06d}.jpg",
                ),
                annotated,
            )

        # ----------------------------------------------------
        # Optional Video 1: test-code diagnostic output.
        #
        # Male cane       = light orange
        # First-year wood = light green
        # Stable/pending winter buds keep B/P identity labels.
        # ----------------------------------------------------
        if SAVE_OUTPUT_VIDEOS:
            diagnostic_frame = apply_colour_coded_masks(
                frame,
                male_mask,
                first_year_mask,
            )

            draw_current_buds_with_ids(
                diagnostic_frame,
                visible_confirmed_buds,
                pending_tracks,
                bud_detections,
                frame_idx,
            )

            draw_frame_summary(
                diagnostic_frame,
                bud_visible=len(visible_confirmed_buds),
                bud_total=bud_total_unique_count,
                frame_id=frame_idx,
            )

            diagnostic_writer.write(diagnostic_frame)

            # ------------------------------------------------
            # Optional Video 2: test-code bud-only output.
            #
            # Original unhighlighted frame with B/P bud IDs
            # and the same winter-bud/frame-ID summary.
            # ------------------------------------------------
            bud_only_frame = frame.copy()

            draw_current_buds_with_ids(
                bud_only_frame,
                visible_confirmed_buds,
                pending_tracks,
                bud_detections,
                frame_idx,
            )

            draw_frame_summary(
                bud_only_frame,
                bud_visible=len(visible_confirmed_buds),
                bud_total=bud_total_unique_count,
                frame_id=frame_idx,
            )

            bud_only_writer.write(bud_only_frame)

        drawing_ms = elapsed_ms(stage_start)

        # ----------------------------------------------------
        # I) CSV ROW -- STRUCTURE KEPT UNCHANGED
        # ----------------------------------------------------
        frame_rows.append({
            "frame_idx": frame_idx,
            "seconds_elapsed": frame_idx / float(fps),
            "bud_raw_count": bud_raw_count,
            "bud_unique_count_this_frame": bud_new_count,
            "bud_old_count_this_frame": bud_old_count,
            "bud_total_unique_count": bud_total_unique_count,
            "bud_final_unique_count_this_frame": bud_new_count,
            "bud_final_total_unique_count": bud_total_unique_count,
            "spur_raw_count": 0,
            "spur_unique_count_this_frame": 0,
            "spur_old_count_this_frame": 0,
            "spur_total_unique_count": 0,
            "lk_dx_px": lk_dx_px,
            "lk_dy_px": lk_dy_px,
            "lk_shift_px": lk_shift_px,
            "lk_tracks_used": lk_tracks_used,
            "lk_ok": lk_ok,
            "raw_count": bud_raw_count,
            "unique_count_this_frame": bud_new_count,
            "total_unique_count": bud_total_unique_count,
        })

        previous_regional_gray = current_regional_gray
        previous_regional_corridor_mask = regional_corridor_mask

        total_frame_ms = elapsed_ms(frame_total_start)
        timing_rows.append({
            "frame_idx": frame_idx,
            "included_in_summary": frame_idx >= PROFILE_WARMUP_FRAMES,
            "male_inference_ms": male_inference_ms,
            "cane_inference_ms": cane_inference_ms,
            "cutie_tracking_ms": cutie_tracking_ms,
            "bud_inference_ms": bud_inference_ms,
            "mask_processing_ms": mask_processing_ms,
            "optical_flow_ms": optical_flow_ms,
            "matching_ms": matching_ms,
            "drawing_ms": drawing_ms,
            "total_frame_ms": total_frame_ms,
        })

        frame_idx += 1
        pbar.update(1)

        del male_results
        del cane_results
        del bud_results
        del cutie_image

finally:
    pbar.close()
    cap.release()

    if diagnostic_writer is not None:
        diagnostic_writer.release()

    if bud_only_writer is not None:
        bud_only_writer.release()


# ============================================================
# 11) SAVE FRAME-ANALYSIS CSV
# ============================================================

FRAME_CSV_COLUMNS = [
    "frame_idx",
    "seconds_elapsed",
    "bud_raw_count",
    "bud_unique_count_this_frame",
    "bud_old_count_this_frame",
    "bud_total_unique_count",
    "bud_final_unique_count_this_frame",
    "bud_final_total_unique_count",
    "spur_raw_count",
    "spur_unique_count_this_frame",
    "spur_old_count_this_frame",
    "spur_total_unique_count",
    "lk_dx_px",
    "lk_dy_px",
    "lk_shift_px",
    "lk_tracks_used",
    "lk_ok",
    "raw_count",
    "unique_count_this_frame",
    "total_unique_count",
]

frame_df = pd.DataFrame(frame_rows, columns=FRAME_CSV_COLUMNS)
frame_df.to_csv(OUTPUT_FRAME_CSV, index=False)

elapsed = time.time() - start_time
final_buds = len(confirmed_tracks)


# ============================================================
# 12) FINAL SUMMARY
# ============================================================

active_buds = sum(
    1 for track in confirmed_tracks.values() if track.get("status") == "active"
)
lost_buds = sum(
    1 for track in confirmed_tracks.values() if track.get("status") == "lost"
)
retired_buds = sum(
    1 for track in confirmed_tracks.values() if track.get("status") == "retired"
)

print("\n================ FINAL SUMMARY ================")
print(
    "Optional diagnostic videos:",
    "enabled" if SAVE_OUTPUT_VIDEOS else "disabled",
)
if SAVE_OUTPUT_VIDEOS:
    print("Cane-highlighted diagnostic video:", OUTPUT_VIDEO)
    print("Bud-only diagnostic video:", OUTPUT_BUD_ONLY_VIDEO)
print("Combined frame CSV:", OUTPUT_FRAME_CSV)
print(
    "Processed frames:",
    PROCESSED_FRAMES_DIR if SAVE_PROCESSED_FRAMES else "disabled",
)
print("Frames processed:", frame_idx)
print("Unique confirmed winter buds:", final_buds)
print("Active bud tracks at end:", active_buds)
print("Lost bud tracks at end:", lost_buds)
print("Retired bud tracks:", retired_buds)
print("Pending unconfirmed buds at end:", len(pending_tracks))
print("Confirmed spurs: 0")
print("Total countable units:", final_buds)
print("Male cane detection instances:", male_detection_instances)
print("First-year wood detection instances:", first_year_detection_instances)
print("Raw winter-bud detection instances:", bud_detection_instances)
print("Bud confirmation:", f"{PENDING_CONFIRM_HITS} of {PENDING_CONFIRM_WINDOW} frames")
print("Bud lost-track memory:", LOST_MAX_GAP_FRAMES, "frames")
print("Regional LK optical flow: ENABLED")
print("Local per-bud optical flow: DISABLED")
print("Hungarian one-to-one matching: ENABLED")
print("Male Cutie processor: ENABLED")
print("First-year Cutie processor: ENABLED")
print("Cutie correction interval:", CUTIE_CORRECTION_INTERVAL, "frames")
print("Male priority on overlap: ENABLED")
print("Elapsed time:", round(elapsed, 2), "seconds")
if frame_idx > 0:
    print("Overall wall-clock seconds/frame:", round(elapsed / frame_idx, 4))
print("================================================")


# ============================================================
# 13) PRIMARY PIPELINE TIMING
# ============================================================

timing_df = pd.DataFrame(timing_rows)

if len(timing_df) > 0:
    summary_df = timing_df[timing_df["included_in_summary"] == True].copy()
    if len(summary_df) == 0:
        summary_df = timing_df.copy()

    mean_total_frame_ms = float(summary_df["total_frame_ms"].mean())

    print(
        "\n========== PRIMARY PIPELINE TIMING "
        f"(warm-up frames excluded: {PROFILE_WARMUP_FRAMES}) =========="
    )
    print(
        f"{'Stage':<28}"
        f"{'Mean ms':>12}"
        f"{'Median':>12}"
        f"{'P95':>12}"
        f"{'% frame':>12}"
    )

    for stage_name in PROFILE_PRIMARY_STAGES:
        values = summary_df[stage_name].astype(float).to_numpy()
        mean_ms = float(np.mean(values))
        median_ms = float(np.median(values))
        p95_ms = safe_percentile(values, 95)
        percent_frame = 100.0 * mean_ms / max(mean_total_frame_ms, 1e-9)

        print(
            f"{stage_name:<28}"
            f"{mean_ms:>12.2f}"
            f"{median_ms:>12.2f}"
            f"{p95_ms:>12.2f}"
            f"{percent_frame:>11.1f}%"
        )

    total_values = summary_df["total_frame_ms"].astype(float).to_numpy()
    print("-" * 76)
    print(
        f"{'total_frame_ms':<28}"
        f"{float(np.mean(total_values)):>12.2f}"
        f"{float(np.median(total_values)):>12.2f}"
        f"{safe_percentile(total_values, 95):>12.2f}"
        f"{100.0:>11.1f}%"
    )

