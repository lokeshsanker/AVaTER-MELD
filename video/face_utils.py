"""
Frame sampling, face detection and alignment for the video branch.

Uses facenet-pytorch MTCNN for face detection + 5-point landmarks.

Pipeline:
    Video frame
        ↓
    MTCNN face detection
        ↓
    Face bounding-box crop
        ↓
    Eye alignment
        ↓
    Square padding
        ↓
    224x224 face image
"""

import cv2
import numpy as np
from facenet_pytorch import MTCNN

from config import NUM_FRAMES, FACE_SIZE, DEVICE


# ---------------------------------------------------------
# MTCNN face detector
# ---------------------------------------------------------

_mtcnn = MTCNN(
    keep_all=False,
    device=DEVICE,
    post_process=False
)


# ---------------------------------------------------------
# Frame sampling
# ---------------------------------------------------------

def sample_frame_indices(
    total_frames: int,
    num_frames: int = NUM_FRAMES
):
    """Uniformly sample num_frames indices across the clip."""

    if total_frames <= 0:
        return []

    if total_frames <= num_frames:
        return list(range(total_frames))

    return np.linspace(
        0,
        total_frames - 1,
        num_frames
    ).astype(int).tolist()


# ---------------------------------------------------------
# Extract raw video frames
# ---------------------------------------------------------

def extract_raw_frames(
    video_path: str,
    num_frames: int = NUM_FRAMES
):
    """Read uniformly sampled RGB frames from a video."""

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    indices = set(
        sample_frame_indices(
            total,
            num_frames
        )
    )

    frames = []

    i = 0

    while cap.isOpened() and len(frames) < num_frames:

        ok, frame = cap.read()

        if not ok:
            break

        if i in indices:

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frames.append(frame)

        i += 1

    cap.release()

    # Pad short videos by repeating last frame
    while (
        len(frames) < num_frames
        and len(frames) > 0
    ):
        frames.append(frames[-1].copy())

    if len(frames) == 0:
        raise RuntimeError(
            f"No frames extracted from: {video_path}"
        )

    return frames


# ---------------------------------------------------------
# Crop + align detected face
# ---------------------------------------------------------

def crop_and_align_face(
    frame: np.ndarray,
    box: np.ndarray,
    landmarks: np.ndarray,
    size: int = FACE_SIZE
):
    """
    Crop the detected face, align using the eye landmarks,
    pad to square and resize to size x size.
    """

    h, w = frame.shape[:2]

    # -----------------------------------------------------
    # Bounding box
    # -----------------------------------------------------

    x1, y1, x2, y2 = box

    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(w, int(np.ceil(x2)))
    y2 = min(h, int(np.ceil(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    # -----------------------------------------------------
    # ACTUAL FACE CROP
    # -----------------------------------------------------

    face_crop = frame[
        y1:y2,
        x1:x2
    ].copy()

    if face_crop.size == 0:
        return None

    # -----------------------------------------------------
    # Convert landmarks from full-frame coordinates
    # to coordinates inside the face crop
    # -----------------------------------------------------

    lm = landmarks.copy()

    lm[:, 0] -= x1
    lm[:, 1] -= y1

    # -----------------------------------------------------
    # Eye landmarks
    #
    # MTCNN landmark order:
    # 0 = left eye
    # 1 = right eye
    # -----------------------------------------------------

    left_eye = lm[0]
    right_eye = lm[1]

    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]

    angle = np.degrees(
        np.arctan2(dy, dx)
    )

    # -----------------------------------------------------
    # Rotate face crop so eyes are horizontal
    # -----------------------------------------------------

    fh, fw = face_crop.shape[:2]

    center = (
        float((left_eye[0] + right_eye[0]) / 2),
        float((left_eye[1] + right_eye[1]) / 2)
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        face_crop,
        rotation_matrix,
        (fw, fh),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # -----------------------------------------------------
    # Pad rotated face to square
    # -----------------------------------------------------

    rh, rw = rotated.shape[:2]

    side = max(rh, rw)

    pad_top = (side - rh) // 2
    pad_bottom = side - rh - pad_top

    pad_left = (side - rw) // 2
    pad_right = side - rw - pad_left

    squared = cv2.copyMakeBorder(
        rotated,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REPLICATE
    )

    # -----------------------------------------------------
    # Resize to CNN input size
    # -----------------------------------------------------

    face = cv2.resize(
        squared,
        (size, size),
        interpolation=cv2.INTER_AREA
    )

    return face.astype(np.uint8)


# ---------------------------------------------------------
# Extract aligned faces from video
# ---------------------------------------------------------

def extract_aligned_faces(
    video_path: str,
    num_frames: int = NUM_FRAMES,
    size: int = FACE_SIZE
):
    """
    Returns:

        (num_frames, size, size, 3)

    uint8 face-crop array.

    If MTCNN fails on a frame, the last successfully
    detected face is reused.
    """

    raw_frames = extract_raw_frames(
        video_path,
        num_frames
    )

    faces = []

    last_good = None

    for frame in raw_frames:

        boxes, probs, landmarks = _mtcnn.detect(
            frame,
            landmarks=True
        )

        face = None

        if (
            boxes is not None
            and landmarks is not None
            and len(boxes) > 0
        ):

            # Select the first detected face
            box = boxes[0]
            lm = landmarks[0]

            face = crop_and_align_face(
                frame,
                box,
                lm,
                size
            )

        # -------------------------------------------------
        # Detection failed
        # -------------------------------------------------

        if face is None:

            if last_good is not None:

                face = last_good.copy()

            else:

                # Only used if no face was detected
                # anywhere in this clip.
                face = cv2.resize(
                    frame,
                    (size, size),
                    interpolation=cv2.INTER_AREA
                )

        last_good = face

        faces.append(face)

    return np.stack(
        faces,
        axis=0
    ).astype(np.uint8)