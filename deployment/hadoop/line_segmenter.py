import cv2
import numpy as np

TARGET_HEIGHT = 128
TARGET_WIDTH = 2048
PAD_VALUE = 255  # white background


def _standardize_and_pad(img: np.ndarray) -> np.ndarray:
    """Resize to TARGET_HEIGHT preserving aspect ratio, then left-pad to TARGET_WIDTH.
    Padding is on the LEFT because Urdu is RTL (text anchors to the right edge).
    """
    h, w = img.shape[:2]
    scale = TARGET_HEIGHT / h
    new_w = max(1, int(w * scale))
    resized = cv2.resize(img, (new_w, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)

    if new_w >= TARGET_WIDTH:
        start = (new_w - TARGET_WIDTH) // 2
        return resized[:, start:start + TARGET_WIDTH]

    padded = np.full((TARGET_HEIGHT, TARGET_WIDTH), PAD_VALUE, dtype=np.uint8)
    padded[:, TARGET_WIDTH - new_w:] = resized  # left-pad (RTL)
    return padded


def _trim_horizontal(crop: np.ndarray, pad_px: int = 12) -> np.ndarray:
    """Trim left/right whitespace from a line crop so the resize step preserves
    the same text-to-canvas ratio the model saw during training.
    """
    _, t = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    col_proj = np.sum(t, axis=0)
    cols = np.where(col_proj > 0)[0]
    if cols.size == 0:
        return crop
    x1 = max(0, cols[0] - pad_px)
    x2 = min(crop.shape[1], cols[-1] + 1 + pad_px)
    return crop[:, x1:x2]


def segment_lines(image_path):
    """
    Reads an image from disk, segments it into individual text lines using a
    horizontal projection profile, then resizes each line to the model's expected
    input size (128 x 2048, grayscale, left-padded for RTL Urdu).

    Floating diacritics (Urdu nuqtas that sit above the main body with a visual
    gap) are merged into the line below them rather than being dropped — losing
    a dot can change ب → ت → ث and ruin OCR.

    Returns:
        List of standardized line images (numpy arrays), each shape (128, 2048).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Images shorter than 64px can't contain more than one text line — skip
    # segmentation entirely so small thumbnails aren't over-split.
    if img.shape[0] < 64:
        return [_standardize_and_pad(img)]

    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(thresh, axis=1)

    # Treat any row with ink as "in-line"; we'll group fragments afterwards.
    ink = proj > 0
    H = img.shape[0]

    # Collect raw runs of ink rows (no height filter yet).
    raw_runs = []
    y = 0
    while y < H:
        if ink[y]:
            s = y
            while y < H and ink[y]:
                y += 1
            raw_runs.append((s, y))
        else:
            y += 1

    if not raw_runs:
        return [_standardize_and_pad(img)]

    # Classify runs as "bodies" (real text lines) vs "fragments" (floating
    # diacritics that ended up in their own run). A run is a body if its height
    # is at least 40% of the tallest run.
    heights = [e - s for s, e in raw_runs]
    body_thresh = max(20, int(max(heights) * 0.4))
    bodies = [list(r) for r in raw_runs if (r[1] - r[0]) >= body_thresh]
    fragments = [r for r in raw_runs if (r[1] - r[0]) < body_thresh]

    if not bodies:
        # No clear text bodies (e.g. very short input) — fall back to raw runs.
        bodies = [list(r) for r in raw_runs]


    # Attach each fragment to the nearest body. In Urdu/Nastaleeq, floating
    # nuqtas typically sit *above* the line they belong to, so on ties prefer
    # the body below.
    for fs, fe in fragments:
        best_idx, best_dist = None, None
        for i, (bs, be) in enumerate(bodies):
            dist_below = bs - fe  # fragment above this body
            dist_above = fs - be  # fragment below this body
            if dist_below >= 0:
                d = (dist_below, 0)  # tie-break: prefer below
            elif dist_above >= 0:
                d = (dist_above, 1)
            else:
                continue  # overlapping; shouldn't happen with raw runs
            if best_dist is None or d < best_dist:
                best_dist, best_idx = d, i
        if best_idx is not None:
            bodies[best_idx][0] = min(bodies[best_idx][0], fs)
            bodies[best_idx][1] = max(bodies[best_idx][1], fe)

    bodies.sort()
    # Pad each line by 20% of its height so the text sits with natural margins,
    # matching the pre-rendered MMU-OCR-21 training images which have whitespace
    # above and below each line.  Cap each edge at the midpoint of the gap to
    # the neighbouring line so crops never steal pixels from adjacent lines.
    n = len(bodies)
    line_regions = []
    for i, (s, e) in enumerate(bodies):
        desired_pad = max(15, int((e - s) * 0.20))

        # How far can we expand upward before hitting the previous line (or image top)?
        if i == 0:
            max_up = s  # can use the full space above
        else:
            gap_above = s - bodies[i - 1][1]
            max_up = gap_above // 2  # take at most half the gap

        # How far can we expand downward before hitting the next line (or image bottom)?
        if i == n - 1:
            max_down = H - e  # can use the full space below
        else:
            gap_below = bodies[i + 1][0] - e
            max_down = gap_below // 2

        pad_up = min(desired_pad, max_up)
        pad_down = min(desired_pad, max_down)
        line_regions.append((max(0, s - pad_up), min(H, e + pad_down)))

    if not line_regions:
        return [_standardize_and_pad(img)]

    line_images = []
    for (y1, y2) in line_regions:
        crop = img[y1:y2, :]
        crop = _trim_horizontal(crop)
        line_images.append(_standardize_and_pad(crop))

    return line_images
