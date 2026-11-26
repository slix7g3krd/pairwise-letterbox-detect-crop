"""Letterbox (black-bar) detection, per-frame tracking and cropping.

The module addresses the following failure modes:

1. **Brightness hysteresis** - a row/column is classified against two
   thresholds (``dark_enter`` / ``dark_exit``).  Values between the thresholds
   keep their previous state, so compression noise and a slowly fading caption
   cannot toggle a bar on and off every frame.
2. **Full-frame detection** - all statistics are computed from the pixels of
   the full-resolution frame.  No thumbnail is used, so a 4 px bar in 1080p
   can no longer be rounded away or shifted by a resize step.
3. **Per-frame tracking** - the crop rectangle is re-evaluated for every
   frame.  New (larger) bars are adopted immediately; shrinking bars require
   ``shrink_confirm`` consecutive frames, so a one-frame flash does not move
   the crop.  A caption strip with bright glyphs is not uniform black and is
   therefore never accepted as a bar.
4. **Even dimensions** - the crop is snapped to even width/height by trimming
   at most one content pixel.  Nothing is padded, so an encoder cannot add a
   black line back into the output.
5. **Display-space cropping** - frames carrying rotation metadata are rotated
   into their display orientation *before* detection and cropping.  Helpers
   map rectangles between storage and display space, and rectangles are always
   clamped inside the frame, never outside it.
6. **Result review** - :func:`review_result` re-inspects the cropped frame:
   no residual bars, even dimensions, rectangle inside the source, rectangle
   matches the detection and the caption-protection decision is honoured.

Frames are ``numpy`` arrays (``uint8`` BGR or grayscale, ``H x W`` /
``H x W x 3``).  Rectangles are ``(x, y, w, h)`` integer tuples in pixels.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

try:  # OpenCV is optional: pure numpy fallbacks are provided for most paths.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - environment dependent
    cv2 = None


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DEFAULTS = {
    # Hysteresis thresholds on the 0..255 luminance scale.  Real letterbox
    # bars are encoded black (0 in full range, ~16 in limited range); a dim
    # night scene sits above that and must not be mistaken for a bar.
    "dark_enter": 20.0,     # at or below this a row may *become* a bar
    "dark_exit": 50.0,      # above this a row stops being a bar
    # a uniform black bar contains almost no bright pixels and is mostly dark
    "bright_level": 110.0,  # pixel above this counts as "bright"
    "max_bright_frac": 0.005,   # 0.5 % bright pixels -> caption, not a bar
    "caption_bright_max": 0.55,  # a caption line still has mostly dark pixels
    "caption_dark_min": 0.45,    # ... but at least this many
    "bar_std_max": 12.0,        # a bar is flat; a night scene is not
    # a real bar has a hard step at its pixel boundary, a fade does not
    "edge_step": 24.0,          # luminance jump considered "hard"
    "edge_frac": 0.85,          # fraction of the boundary that must be hard
    # sanity limits
    "max_bar_frac": 0.40,   # a bar can never cover more than 40 % of an axis
    "min_content_frac": 0.30,  # and at least 30 % of the axis must remain
    # tracking
    "shrink_confirm": 3,
    # review tolerance in pixels
    "residual_tol": 1,
}


def _params(overrides=None):
    """Return a fresh parameter dictionary, validated against ``DEFAULTS``."""

    p = dict(DEFAULTS)
    if overrides:
        for key, value in dict(overrides).items():
            if key in DEFAULTS:
                p[key] = value
    if p["dark_enter"] > p["dark_exit"]:
        p["dark_enter"], p["dark_exit"] = p["dark_exit"], p["dark_enter"]
    return p


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_frame(value):
    """True when *value* looks like one image frame."""

    if not isinstance(value, np.ndarray):
        return False
    if value.size == 0 or value.ndim not in (2, 3):
        return False
    if value.ndim == 3 and value.shape[2] not in (1, 3, 4):
        return False
    return True


def _coerce_frame(value):
    """Coerce numpy array-likes, nested lists or PIL images into a frame.

    Returns ``None`` when the value cannot represent a 2D/3D image.  Already
    valid numpy frames are returned unchanged.
    """

    if _is_frame(value):
        return value
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return None
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.dtype == object:
        try:
            arr = np.stack([np.asarray(row) for row in value])
        except Exception:
            return None
    if arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (1, 3, 4)):
        return arr
    return None


def _frame_list(value):
    """Normalise an arbitrary input into a list of frames."""

    if _is_frame(value):
        return [value]
    if isinstance(value, np.ndarray) and value.ndim >= 3:
        # H x W x C is a single frame and was handled above.  Any other 3D
        # array is an N x H x W grayscale stack; a 4D array is N x H x W x C.
        is_hwc = value.ndim == 3 and value.shape[2] in (1, 3, 4)
        if not is_hwc and value.ndim in (3, 4):
            return [value[i] for i in range(value.shape[0])]
        return []
    if isinstance(value, (list, tuple)):
        frames = []
        for item in value:
            frame = _coerce_frame(item)
            if frame is not None:
                frames.append(frame)
        if frames:
            return frames
        # A single frame supplied as a nested Python list/tuple of rows.
        whole = _coerce_frame(value)
        return [whole] if whole is not None else []
    return []


def _to_luminance(frame):
    """Return a float32 luminance plane (0..255) or ``None``."""

    if not _is_frame(frame):
        return None
    arr = frame
    if arr.ndim == 3:
        channels = arr.shape[2]
        if channels == 1:
            arr = arr[..., 0]
        elif channels >= 3:
            if arr.dtype == np.uint8 and cv2 is not None:
                arr = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                # ITU-R BT.601 weights, BGR order
                b = arr[..., 0].astype(np.float32)
                g = arr[..., 1].astype(np.float32)
                r = arr[..., 2].astype(np.float32)
                arr = 0.114 * b + 0.587 * g + 0.299 * r
        else:  # pragma: no cover - defensive
            arr = arr[..., 0]
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr.astype(np.float32)
    if arr.dtype == bool:
        return arr.astype(np.float32) * 255.0
    if np.issubdtype(arr.dtype, np.integer):
        return np.clip(arr, 0, 255).astype(np.float32)
    arr = arr.astype(np.float32)
    if arr.size and float(np.nanmax(arr)) <= 1.0:
        arr = arr * 255.0
    return np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)


def _axis_stats(lum, axis, params):
    """Mean / std / median / peak / bright / dark fractions along ``axis``.

    ``median`` is the *background* brightness: sparse subtitle glyphs drawn on
    a black bar barely move it, whereas they drag the mean up.  ``peak`` (99th
    percentile) tells whether the line is uniformly near-black or contains
    structure.  Together they separate "text on a bar" (low background, high
    peak) from "a dim picture line" (higher background).
    """

    return {
        "mean": lum.mean(axis=axis),
        "median": np.median(lum, axis=axis),
        "peak": np.percentile(lum, 99, axis=axis),
        "std": lum.std(axis=axis),
        "bright": (lum > params["bright_level"]).mean(axis=axis),
        "dark": (lum <= params["dark_exit"]).mean(axis=axis),
    }


def _uniform_mask(stats, params):
    """Per-line mask: is this line consistent with a uniform black bar?"""

    return (
        (stats["std"] <= params["bar_std_max"])
        & (stats["bright"] <= params["max_bright_frac"])
        & (stats["median"] <= params["dark_enter"])
    )


def _empty_detection(params=None, shape=(0, 0), rotation=0):
    p = params or _params()
    return {
        "bars": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        "rect": (0, 0, 0, 0),
        "even_rect": (0, 0, 0, 0),
        "frame_shape": tuple(shape),
        "display_shape": tuple(shape),
        "storage_shape": tuple(shape),
        "rotation": int(rotation),
        "space": "display",
        "suspect": True,
        "caption_protected": False,
        "protected_sides": [],
        "flags": {"top": "invalid", "bottom": "invalid",
                  "left": "invalid", "right": "invalid"},
        "content_loss": {"right": 0, "bottom": 0},
        "rect_outside": False,
        "detection_scale": 1.0,
        "thumbnail": False,
        "ok": False,
        "params": p,
    }


def _normalise_rect(rect, shape):
    """Coerce a rectangle to four ints clamped inside ``shape``."""

    H, W = int(shape[0]), int(shape[1])
    try:
        x, y, w, h = (int(round(float(v))) for v in rect)
    except Exception:
        return (0, 0, 0, 0), True
    clamped = False
    if x < 0:
        w += x
        x = 0
        clamped = True
    if y < 0:
        h += y
        y = 0
        clamped = True
    if w <= 0 or h <= 0:
        return (min(max(x, 0), W), min(max(y, 0), H), 0, 0), True
    if x + w > W:
        w = W - x
        clamped = True
    if y + h > H:
        h = H - y
        clamped = True
    return (x, y, w, h), clamped


def _align_even_rect(rect, shape):
    """Snap a rectangle to even width/height by trimming content pixels.

    At most one content pixel is dropped on each axis.  Black pixels are
    never padded back into the rectangle.
    """

    x, y, w, h = _normalise_rect(rect, shape)[0]
    loss_x = 0
    loss_y = 0
    if w % 2:
        if w >= 2:
            w -= 1
            loss_x = 1
        else:  # pragma: no cover - degenerate
            w = 0
    if h % 2:
        if h >= 2:
            h -= 1
            loss_y = 1
        else:  # pragma: no cover - degenerate
            h = 0
    if w and x + w > int(shape[1]):
        x = int(shape[1]) - w
    if h and y + h > int(shape[0]):
        y = int(shape[0]) - h
    return (x, y, w, h), (loss_x, loss_y)


def _display_shape(shape, rotation):
    """Shape (H, W) of a frame after applying metadata ``rotation``."""

    H, W = int(shape[0]), int(shape[1])
    rotation = _normalise_rotation(rotation)
    if rotation in (90, 270):
        return (W, H)
    return (H, W)


def _normalise_rotation(rotation):
    try:
        rotation = int(round(float(rotation)))
    except Exception:
        return 0
    rotation %= 360
    # snap to the nearest right angle
    if rotation in (0, 90, 180, 270):
        return rotation
    return int(round(rotation / 90.0)) * 90 % 360


# ---------------------------------------------------------------------------
# 1. Brightness hysteresis
# ---------------------------------------------------------------------------

def brightness_hysteresis(values, dark_enter=None, dark_exit=None,
                          initial=None, as_array=True, **kwargs):
    """Classify luminance values as dark (bar) / light (content) with hysteresis.

    A value at or below ``dark_enter`` switches a line *to* dark; only a value
    above ``dark_exit`` switches it back to light.  Anything in between keeps
    its previous state.  The result is a boolean array (``True`` = dark).

    Common alternative keyword names (``enter``/``exit``, ``low``/``high``,
    ``on``/``off`` ...) are accepted as aliases for the two thresholds.
    """

    if dark_enter is None:
        for key in ("enter", "low", "threshold_enter", "dark", "on", "lo"):
            if key in kwargs:
                dark_enter = kwargs[key]
                break
    if dark_exit is None:
        for key in ("exit", "high", "threshold_exit", "light", "off", "hi"):
            if key in kwargs:
                dark_exit = kwargs[key]
                break

    p = _params()
    enter = p["dark_enter"] if dark_enter is None else float(dark_enter)
    exit_ = p["dark_exit"] if dark_exit is None else float(dark_exit)
    if enter > exit_:
        enter, exit_ = exit_, enter
    try:
        arr = np.asarray(values, dtype=np.float64).ravel()
    except Exception:
        arr = np.asarray([], dtype=np.float64)
    if arr.size == 0:
        out = np.zeros(0, dtype=bool)
        return out if as_array else out.tolist()
    if initial is None:
        state = bool(arr[0] <= enter)
    else:
        state = bool(initial)
    out = np.empty(arr.size, dtype=bool)
    for i, value in enumerate(arr):
        value = float(value)
        if state:
            if value > exit_:
                state = False
                out[i] = False
                continue
        else:
            if value <= enter:
                state = True
                out[i] = True
                continue
            out[i] = False
            continue
        out[i] = True
    return out if as_array else out.tolist()


# ---------------------------------------------------------------------------
# 2. Full-frame detection
# ---------------------------------------------------------------------------

def _line_uniformity(stats, params, allow_text=True):
    """Classify lines: 0 = content, 1 = pure bar, 2 = bar with sparse text.

    A pure bar is flat black.  A subtitle rendered on top of a black bar adds
    sparse bright glyphs; the *background* (median) stays at bar level and the
    line is still mostly dark, so the line is still *inside* the bar.  Cropping
    here would cut the text in half, so it must be treated as a protected bar
    line.  A dim picture line has an elevated background/peak and ends the run.

    ``allow_text`` is used for top/bottom bars.  On a column profile the black
    parts of the top and bottom bars look deceptively like sparse text, so for
    left/right bars a line is either pure black or content; the scan then stops
    before any glyphs and can never cut through them.
    """

    bright = stats["bright"]
    dark = stats["dark"]
    std = stats["std"]
    median = stats["median"]
    peak = stats["peak"]

    pure = (
        (bright <= params["max_bright_frac"])
        & (std <= params["bar_std_max"])
        & (median <= params["dark_enter"])
    )
    kind = np.zeros(bright.shape, dtype=np.int8)
    kind[pure] = 1
    if allow_text:
        text = (
            (bright > params["max_bright_frac"])
            & (median <= params["dark_enter"])
            & (dark >= params["caption_dark_min"])
            & (bright <= params["caption_bright_max"])
            & (peak >= params["bright_level"])
        )
        kind[text] = 2
    return kind


def _scan_contiguous_bar(background, kinds, params):
    """Length of the bar connected to the edge, plus whether it holds text.

    ``background`` is the per-line *median* luminance, so sparse subtitle
    glyphs do not lift a line out of the bar.  Lines are visited from the frame
    edge inwards; the run ends at the first line classified as picture content
    or whose background exceeds the hysteresis exit threshold.  The hysteresis
    state starts at "bar", so a gradual fade inside a real bar is not mistaken
    for its boundary.
    """

    if len(background) == 0:
        return 0, False
    states = brightness_hysteresis(
        background, params["dark_enter"], params["dark_exit"], initial=True
    )
    length = 0
    has_text = False
    for i in range(len(background)):
        if kinds[i] == 0 or not states[i]:
            break
        length = i + 1
        if kinds[i] == 2:
            has_text = True
    return length, has_text


def _has_hard_edge(profile, boundary, params, band=None):
    """Does the bar/content boundary look like a real letterbox edge?

    A genuine letterbox edge is a step: nearly every pixel along the boundary
    changes abruptly from black to picture.  A dim night scene that fades to
    black has no such step, so this check stops a dark picture from being
    mistaken for bars.

    ``band`` restricts the measurement to ``(start, stop)`` along the axis
    parallel to the boundary.  Bars on the perpendicular sides are black on
    both sides of this boundary, so including them would dilute the step
    fraction and make a real bar look soft; the caller passes the content band
    to exclude them.  Lines that hold subtitles are exempt from this check.
    """

    H = profile.shape[0]
    if boundary <= 0 or boundary >= H:
        return False
    if band is not None:
        start, stop = (max(0, int(band[0])), min(profile.shape[1], int(band[1])))
        if stop - start >= 8:
            profile = profile[:, start:stop]
    above = profile[max(0, boundary - 3):boundary].mean(axis=0)
    below = profile[boundary:min(H, boundary + 3)].mean(axis=0)
    step = below - above
    frac = float((step >= params["edge_step"]).mean())
    return frac >= params["edge_frac"]


def _side_bar(background, kinds, params, axis_len, side):
    """Raw bar length (no cap) plus caption-safety decision for one side."""

    length, has_text = _scan_contiguous_bar(background, kinds, params)
    flag = "bar" if length else "none"
    protected = False

    if has_text:
        # A caption lives inside this bar.  Cropping the bar would cut the
        # caption in half, so the safe decision is not to crop this side.
        protected = True
        flag = "caption"
        length = 0
    return length, flag, protected, has_text


def _build_detection(shape, bars, flags, params, protected_sides, shape_storage=None,
                     rotation=0):
    H, W = int(shape[0]), int(shape[1])
    top, bottom = int(bars["top"]), int(bars["bottom"])
    left, right = int(bars["left"]), int(bars["right"])
    rect = (left, top, W - left - right, H - top - bottom)
    even_rect, loss = _align_even_rect(rect, (H, W))

    suspect = False
    if rect[2] <= 0 or rect[3] <= 0:
        suspect = True
    min_w = params["min_content_frac"] * W
    min_h = params["min_content_frac"] * H
    if (W - left - right) < min_w or (H - top - bottom) < min_h:
        # Detection collapsed: rather than emit a bogus sliver, fall back to
        # the full frame and mark the result as suspect.
        bars = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        flags = {side: "invalid" for side in flags}
        rect = (0, 0, W, H)
        even_rect, loss = _align_even_rect(rect, (H, W))
        suspect = True

    return {
        "bars": bars,
        "rect": rect,
        "even_rect": even_rect,
        "frame_shape": (H, W),
        "display_shape": (H, W),
        "storage_shape": shape_storage or (H, W),
        "rotation": int(rotation),
        "space": "display",
        "suspect": suspect,
        "caption_protected": bool(protected_sides),
        "protected_sides": list(protected_sides),
        "flags": flags,
        "content_loss": {"right": loss[0], "bottom": loss[1]},
        "rect_outside": False,
        "detection_scale": 1.0,
        "thumbnail": False,
        "ok": (not suspect) and rect[2] > 0 and rect[3] > 0,
        "params": params,
    }


def detect_full_frame(frame, rotation=0, params=None, **kwargs):
    """Detect bars on a single frame at full resolution.

    The frame is first brought into display orientation with ``rotation``
    (degrees counter-clockwise, matching :func:`apply_rotation`); every pixel
    of the resulting frame participates in the statistics.
    """

    p = _params({**dict(params or {}), **kwargs})
    frame = _coerce_frame(frame)
    if not _is_frame(frame):
        return _empty_detection(p)

    rotation = _normalise_rotation(rotation)
    display = apply_rotation(frame, rotation)
    lum = _to_luminance(display)
    if lum is None:
        return _empty_detection(p, shape=tuple(display.shape[:2]), rotation=rotation)

    shapes = np.array(display.shape[:2])
    rows = _axis_stats(lum, 1, p)
    cols = _axis_stats(lum, 0, p)
    row_kinds = _line_uniformity(rows, p, allow_text=True)
    col_kinds = _line_uniformity(cols, p, allow_text=False)

    H, W = int(shapes[0]), int(shapes[1])
    bars = {}
    flags = {}
    protected = []
    profiles = {}

    side_specs = (
        ("top", rows["median"], row_kinds, H, lum),
        ("bottom", rows["median"][::-1], row_kinds[::-1], H, lum[::-1]),
        ("left", cols["median"], col_kinds, W, lum.T),
        ("right", cols["median"][::-1], col_kinds[::-1], W, lum.T[::-1]),
    )

    # First pass: raw contiguous-bar lengths per side (no hard-edge test yet,
    # and deliberately not capped).  The hard-edge test needs the perpendicular
    # bar lengths, so it runs only after every side has been scanned.
    raw = {}
    for side, background, kinds, axis_len, profile in side_specs:
        length, flag, was_protected, had_text = _side_bar(
            background, kinds, p, axis_len, side
        )
        raw[side] = length
        flags[side] = flag
        if was_protected:
            protected.append(side)
        profiles[side] = np.asarray(profile, dtype=np.float32)

    # Second pass: validate the edge and enforce the sanity limit.
    for side, background, kinds, axis_len, profile in side_specs:
        length = raw[side]
        if length:
            if side in ("top", "bottom"):
                # Exclude the perpendicular bars from the step measurement:
                # their black pixels lie on both sides of this boundary and
                # would otherwise dilute a genuine edge.
                band = (raw["left"], W - raw["right"])
            else:
                band = (raw["top"], H - raw["bottom"])
            if not _has_hard_edge(profiles[side], length, p, band=band):
                # A dark picture that fades to black: not a letterbox bar.
                length = 0
                flags[side] = "no_edge"
        if length > int(p["max_bar_frac"] * axis_len):
            # Beyond the sanity limit this is more likely a dark picture than
            # a bar.  Never truncate a bar part-way (that cuts into the bar and
            # leaves black exposed): leave the side uncropped instead.
            length = 0
            flags[side] = "capped"
        bars[side] = length
        if flags[side] == "caption" and side not in protected:
            # A caption lives inside this bar.  Cropping it would cut the text
            # in half, so leave the side uncropped and remember to protect it.
            protected.append(side)

    return _build_detection(
        (H, W), bars, flags, p, protected,
        shape_storage=tuple(frame.shape[:2]), rotation=rotation,
    )


# ---------------------------------------------------------------------------
# 3. Per-frame tracking
# ---------------------------------------------------------------------------

def _bars_to_rect(bars, shape, params):
    H, W = int(shape[0]), int(shape[1])
    top = max(0, min(int(bars["top"]), H))
    bottom = max(0, min(int(bars["bottom"]), H))
    left = max(0, min(int(bars["left"]), W))
    right = max(0, min(int(bars["right"]), W))
    if top + bottom >= H:
        top = bottom = 0
    if left + right >= W:
        left = right = 0
    rect = (left, top, W - left - right, H - top - bottom)
    return _align_even_rect(rect, (H, W))


def track_per_frame(frames, rotation=0, params=None, detections=None, **kwargs):
    """Track the content rectangle across a sequence, one detection per frame.

    * Growing bars (camera zoom-out exposes new black) are adopted at once.
    * Shrinking bars must be confirmed for ``shrink_confirm`` consecutive
      frames so a single flash frame does not move the crop.
    * A caption inside a previously cropped bar is honoured immediately, since
      cropping it would damage the subtitles.
    """

    p = _params({**dict(params or {}), **kwargs})
    frames = _frame_list(frames)
    rotation = _normalise_rotation(rotation)

    if detections is None:
        detections = [
            detect_full_frame(f, rotation=rotation, params=p) for f in frames
        ]
    detections = [d for d in detections if isinstance(d, dict) and "bars" in d]

    sides = ("top", "bottom", "left", "right")
    current = None
    pending = {side: 0 for side in sides}
    rects = []
    bar_history = []
    shapes = []

    for det in detections:
        cand = {side: int(det["bars"].get(side, 0)) for side in sides}
        protected = set(det.get("protected_sides") or ())
        if current is None:
            current = dict(cand)
        else:
            for side in sides:
                if cand[side] > current[side]:
                    current[side] = cand[side]
                    pending[side] = 0
                elif cand[side] < current[side]:
                    if side in protected:
                        current[side] = cand[side]
                        pending[side] = 0
                    else:
                        pending[side] += 1
                        if pending[side] >= p["shrink_confirm"]:
                            current[side] = cand[side]
                            pending[side] = 0
                else:
                    pending[side] = 0

        shape = tuple(det.get("display_shape") or det.get("frame_shape") or (0, 0))
        rect, loss = _bars_to_rect(current, shape, p)
        rects.append(rect)
        bar_history.append(dict(current))
        shapes.append(shape)

    return {
        "frames": len(rects),
        "rects": rects,
        "bars": bar_history,
        "detections": detections,
        "rotation": rotation,
        "display_shapes": shapes,
        "shrink_pending": dict(pending),
        "params": p,
        "ok": bool(rects),
    }


def detect(source, rotation=0, params=None, **kwargs):
    """Detect on one frame, or track across a sequence of frames.

    ``source`` may be a frame, a list/array of frames, or a bundle returned by
    :func:`load` (in which case its rotation metadata is used).
    """

    if isinstance(source, dict) and "frames" in source:
        frames = _frame_list(source.get("frames"))
        if rotation is None or rotation == 0:
            rotation = source.get("rotation", 0)
        if len(frames) == 1:
            return detect_full_frame(frames[0], rotation=rotation,
                                     params=params, **kwargs)
        return track_per_frame(frames, rotation=rotation, params=params, **kwargs)
    if _is_frame(source):
        return detect_full_frame(source, rotation=rotation, params=params, **kwargs)
    return track_per_frame(source, rotation=rotation, params=params, **kwargs)


# ---------------------------------------------------------------------------
# 4. Even dimensions
# ---------------------------------------------------------------------------

def _even_int(value):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    if n <= 0:
        return 0
    return n - (n % 2)


def even_dimension(*args, **kwargs):
    """Even size / rectangle helpers.

    * ``even_dimension(n)`` -> largest even integer <= ``n``.
    * ``even_dimension(w, h)`` or ``even_dimension((w, h))`` ->
      ``(even_w, even_h)`` (trim, never pad).
    * ``even_dimension((x, y, w, h))`` -> the rectangle with even ``w``/``h``.
    * ``even_dimension((x, y, w, h), (H, W))`` -> rectangle trimmed to even
      ``w``/``h`` and clamped to ``shape`` (never padded).
    """

    # even_dimension(w, h) with two scalar sizes.
    if len(args) == 2 and all(
        not isinstance(v, (tuple, list, np.ndarray)) for v in args
    ):
        return (_even_int(args[0]), _even_int(args[1]))

    # even_dimension((w, h)) / even_dimension((x, y, w, h)).
    if len(args) == 1 and isinstance(args[0], (tuple, list, np.ndarray)):
        seq = np.asarray(args[0])
        if seq.ndim == 1 and seq.shape[0] == 2:
            return (_even_int(seq[0]), _even_int(seq[1]))
        if seq.ndim == 1 and seq.shape[0] == 4:
            return (int(seq[0]), int(seq[1]), _even_int(seq[2]),
                    _even_int(seq[3]))
        if isinstance(args[0], (tuple, list)) and len(args[0]) == 3:
            # A (H, W, C) shape tuple: align width and height.
            try:
                return (_even_int(args[0][1]), _even_int(args[0][0]))
            except (TypeError, ValueError, IndexError):
                return 0
        return (0, 0, 0, 0)

    if len(args) == 1 and not isinstance(args[0], (tuple, list, np.ndarray)):
        return _even_int(args[0])

    if len(args) >= 2:
        rect, shape = args[0], args[1]
        return _align_even_rect(rect, shape)

    rect = kwargs.get("rect")
    shape = kwargs.get("shape")
    if rect is not None and shape is not None:
        return _align_even_rect(rect, shape)
    if rect is not None:
        return even_dimension(rect)
    return 0


# ---------------------------------------------------------------------------
# 5. Display space: rotation and coordinate mapping
# ---------------------------------------------------------------------------

def apply_rotation(frame, rotation=0):
    """Rotate a frame counter-clockwise into display orientation."""

    if not _is_frame(frame):
        return frame
    k = _normalise_rotation(rotation) // 90
    if k == 0:
        return frame
    return np.ascontiguousarray(np.rot90(frame, k))


def map_rect_between_spaces(rect, shape, rotation=0, to_display=True):
    """Map a rectangle between storage and display space.

    ``shape`` is the shape of the *source* space (storage space when
    ``to_display`` is true, display space otherwise).  The result is clamped
    inside the destination frame so a rectangle can never land outside it.
    """

    H, W = int(shape[0]), int(shape[1])
    r = _normalise_rotation(rotation)
    if not to_display:
        r = (360 - r) % 360
    x, y, w, h = (int(round(float(v))) for v in rect)

    if r == 0:
        mapped = (x, y, w, h)
    elif r == 90:  # CCW
        mapped = (y, W - (x + w), h, w)
    elif r == 180:
        mapped = (W - (x + w), H - (y + h), w, h)
    elif r == 270:  # CW
        mapped = (H - (y + h), x, h, w)
    else:  # pragma: no cover
        mapped = (x, y, w, h)

    dest_shape = _display_shape(shape, r if to_display else (360 - r) % 360)
    return _normalise_rect(mapped, dest_shape)[0]


def crop_in_display_space(frame, rect=None, rotation=0, rect_space="display"):
    """Crop a frame after bringing it into display orientation.

    ``rect`` is interpreted in ``rect_space`` ("display" or "storage").  The
    returned rectangle is reported in display coordinates and cannot lie
    outside the display frame; storage coordinates are provided for callers
    that need to map back.
    """

    result = {
        "frame": None,
        "rect": (0, 0, 0, 0),
        "rect_display": (0, 0, 0, 0),
        "rect_storage": (0, 0, 0, 0),
        "rotation": _normalise_rotation(rotation),
        "display_shape": (0, 0),
        "storage_shape": (0, 0),
        "ok": False,
        "clamped": False,
    }
    if not _is_frame(frame):
        frame = _coerce_frame(frame)
    if not _is_frame(frame):
        return result

    rotation = _normalise_rotation(rotation)
    storage_shape = tuple(int(v) for v in frame.shape[:2])
    display = apply_rotation(frame, rotation)
    display_shape = tuple(int(v) for v in display.shape[:2])
    result.update(
        rotation=rotation,
        display_shape=display_shape,
        storage_shape=storage_shape,
    )

    if rect is None:
        rect_display = (0, 0, display_shape[1], display_shape[0])
    elif rect_space == "storage":
        rect_display = map_rect_between_spaces(
            rect, storage_shape, rotation, to_display=True
        )
    else:
        rect_display, _ = _normalise_rect(rect, display_shape)

    rect_display, clamped = _normalise_rect(rect_display, display_shape)
    rect_storage = map_rect_between_spaces(
        rect_display, display_shape, rotation, to_display=False
    )
    crop = display[
        rect_display[1]:rect_display[1] + rect_display[3],
        rect_display[0]:rect_display[0] + rect_display[2],
    ]
    result.update(
        frame=np.ascontiguousarray(crop),
        rect=rect_display,
        rect_display=rect_display,
        rect_storage=rect_storage,
        clamped=bool(clamped),
        ok=rect_display[2] > 0 and rect_display[3] > 0,
    )
    return result


def crop(frame, *args, rect=None, rotation=0, rect_space="display",
         auto=True, **kwargs):
    """Return the cropped frame (display-space crop, see
    :func:`crop_in_display_space`).

    Accepted calling conventions::

        crop(frame)                          # auto-detect the letterbox rect
        crop(frame, (x, y, w, h))
        crop(frame, x, y, w, h)
        crop(frame, (x, y, w, h), 90)        # rect then rotation
        crop(frame, rect=(x, y, w, h), rotation=90)

    With no rectangle the bars are detected on the full frame and the even
    rectangle is used; pass ``auto=False`` to keep the full frame.
    """

    args = list(args)

    # Old positional signature: crop(frame, rect[, rotation[, rect_space]]).
    if rect is None and args and (
        args[0] is None or isinstance(args[0], (tuple, list, np.ndarray))
    ):
        rect = args.pop(0)

    # Four scalar positionals mean crop(frame, x, y, w, h).
    if rect is None and len(args) >= 4:
        try:
            rect = tuple(int(round(float(v))) for v in args[:4])
        except (TypeError, ValueError):
            rect = None
        args = args[4:]

    # Any remaining positionals are rotation[, rect_space].
    if args:
        if args[0] is not None and not isinstance(
            args[0], (tuple, list, np.ndarray)
        ):
            rotation = args[0]
        if len(args) >= 2 and isinstance(args[1], str):
            rect_space = args[1]

    if rect is not None and not isinstance(rect, (tuple, list, np.ndarray)):
        rect = None

    if rect is None and auto and _is_frame(_coerce_frame(frame)):
        det = detect_full_frame(frame, rotation=rotation, **kwargs)
        if isinstance(det, dict):
            rect = det.get("even_rect") or det.get("rect")

    info = crop_in_display_space(frame, rect=rect, rotation=rotation,
                                 rect_space=rect_space)
    if info["frame"] is None:
        valid = _coerce_frame(frame)
        channels = 3 if _is_frame(valid) and valid.ndim == 3 else 0
        return np.zeros((0, 0, channels), dtype=np.uint8) if channels else \
            np.zeros((0, 0), dtype=np.uint8)
    return info["frame"]


def crop_frames(frames, rects, rotation=0):
    """Crop a sequence; ``rects`` are display-space rectangles."""

    frames = _frame_list(frames)
    rects = list(rects or [])
    out = []
    for i, frame in enumerate(frames):
        rect = rects[i] if i < len(rects) else None
        out.append(crop(frame, rect=rect, rotation=rotation))
    return out


# ---------------------------------------------------------------------------
# 6. Result review / validation / metrics / audit
# -------------------------------------------------------------------------

def _bar_ratio(det):
    shape = det.get("display_shape") or det.get("frame_shape") or (0, 0)
    H, W = shape
    if not H or not W:
        return 0.0
    area = det["bars"]["top"] * W + det["bars"]["bottom"] * W + \
        det["bars"]["left"] * H + det["bars"]["right"] * H
    return float(area) / float(H * W)


def _rect_equal(a, b, tol=1):
    return all(abs(int(a[i]) - int(b[i])) <= tol for i in range(4))


def _even_rect_ok(det):
    rect = det.get("even_rect") or (0, 0, 0, 0)
    return int(rect[2]) % 2 == 0 and int(rect[3]) % 2 == 0


def validate_letterbox(detection):
    """Check internal consistency of a detection (or a list of them)."""

    if isinstance(detection, (list, tuple)):
        items = [validate_letterbox(d) for d in detection]
        return {
            "ok": all(item["ok"] for item in items),
            "issues": [issue for item in items for issue in item["issues"]],
            "count": len(items),
        }
    if not isinstance(detection, dict) or "bars" not in detection:
        return {"ok": False, "issues": ["not a detection dictionary"]}

    issues = []
    bars = detection.get("bars") or {}
    shape = tuple(detection.get("display_shape") or detection.get("frame_shape") or (0, 0))
    H, W = shape
    if H <= 0 or W <= 0:
        return {"ok": False, "issues": ["empty frame shape"]}

    for side in ("top", "bottom", "left", "right"):
        value = int(bars.get(side, -1))
        if value < 0:
            issues.append(f"{side} bar negative")
        if value == 0:
            continue
        limit = (H if side in ("top", "bottom") else W)
        if value > int(0.5 * limit):
            issues.append(f"{side} bar exceeds half the frame")

    rect = tuple(int(v) for v in (detection.get("rect") or (0, 0, 0, 0)))
    even = tuple(int(v) for v in (detection.get("even_rect") or rect))
    if rect[2] <= 0 or rect[3] <= 0:
        issues.append("empty content rectangle")
    if rect[0] < 0 or rect[1] < 0 or rect[0] + rect[2] > W or rect[1] + rect[3] > H:
        issues.append("rectangle outside the display frame")
    if not _rect_equal(rect, (even[0], even[1], even[2], even[3]), tol=2):
        issues.append("even rectangle shifts by more than 2 px")
    if even[2] % 2 or even[3] % 2:
        issues.append("even rectangle is not even")
    if even[2] <= 0 or even[3] <= 0:
        issues.append("empty even rectangle")

    # The rectangle must cover this frame's bars.  A caption-protected side
    # intentionally keeps a bar uncropped, so it is not required to match.
    content = (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3])
    for side, bar in bars.items():
        bar = int(bar)
        if bar <= 0:
            continue
        if side == "top" and content[1] < bar:
            issues.append("rectangle does not cover the top bar")
        elif side == "bottom" and content[3] < H - bar:
            issues.append("rectangle does not cover the bottom bar")
        elif side == "left" and content[0] < bar:
            issues.append("rectangle does not cover the left bar")
        elif side == "right" and content[2] < W - bar:
            issues.append("rectangle does not cover the right bar")

    return {
        "ok": not issues,
        "issues": issues,
        "bars": dict(bars),
        "rect": rect,
        "even_rect": even,
        "display_shape": shape,
        "caption_protected": bool(detection.get("caption_protected")),
        "protected_sides": list(detection.get("protected_sides") or ()),
    }


def _empty_metrics():
    return {"ok": False, "frames": 0, "bar_ratio_mean": 0.0,
            "bar_ratio_max": 0.0, "bar_ratio_min": 0.0,
            "caption_protected_frames": 0, "suspect_frames": 0,
            "detection_scale": 1.0, "all_even": True}


def metrics(source, rects=None, rotation=0, params=None):
    """Summarise detection quality: bar ratios, retention, protections."""

    # A tracking bundle carries a per-frame bar history; a single detection
    # carries one "bars" dictionary.  Check the sequence form first so a
    # bundle is never mistaken for a frame.
    if isinstance(source, dict) and "detections" in source:
        dets = [d for d in (source.get("detections") or []) if isinstance(d, dict)]
        if not dets:
            return _empty_metrics()
        ratios = [_bar_ratio(d) for d in dets]
        return {
            "ok": True,
            "frames": len(dets),
            "bar_ratio_mean": float(np.mean(ratios)),
            "bar_ratio_max": float(np.max(ratios)),
            "bar_ratio_min": float(np.min(ratios)),
            "caption_protected_frames": sum(
                1 for d in dets if d.get("caption_protected")
            ),
            "suspect_frames": sum(1 for d in dets if d.get("suspect")),
            "detection_scale": 1.0,
            "all_even": all(_even_rect_ok(d) for d in dets),
        }

    if isinstance(source, dict) and isinstance(source.get("bars"), dict):
        det = source
        return {
            "ok": bool(det.get("ok", True)),
            "bar_ratio": _bar_ratio(det),
            "bars": dict(det["bars"]),
            "caption_protected": bool(det.get("caption_protected")),
            "suspect": bool(det.get("suspect")),
            "detection_scale": float(det.get("detection_scale", 1.0)),
            "rect_outside": bool(det.get("rect_outside")),
            "even": _even_rect_ok(det),
            "frames": 1,
        }

    frames = _frame_list(source)
    if not frames:
        return _empty_metrics()

    dets = [detect_full_frame(f, rotation=rotation, params=params) for f in frames]
    tracked = track_per_frame(frames, rotation=rotation, params=params,
                              detections=dets)
    result = metrics(tracked)
    if rects is not None:
        checked = [
            _rect_equal(det.get("even_rect") or (0, 0, 0, 0), rect, tol=1)
            for det, rect in zip(dets, list(rects))
        ]
        result["rect_matches_detection"] = bool(checked) and all(checked)
    return result


def audit(frames, rects=None, rotation=0, params=None):
    """Audit the chosen rectangles against a fresh full-frame detection.

    Per frame this looks back and verifies:

    * **errors** — rectangle outside the display frame, odd crop dimensions,
      or a bar that the frame still has but the rectangle does not remove
      (the failure where a zoom-out exposes black on one side).
    * **warnings** — the rectangle removes picture content that this frame's
      detection calls content.  Per-frame tracking deliberately holds the
      previous rectangle for ``shrink_confirm`` frames when a bar appears to
      shrink, so a transient bright frame can be over-cropped slightly.  That
      is bounded and is reported separately from real defects.
    """

    p = _params(params)
    frames = _frame_list(frames)
    rects = list(rects or [])
    if not frames:
        return {"ok": False, "issues": ["no frames"], "warnings": [],
                "frames": 0, "items": []}

    items = []
    all_ok = True
    for i, frame in enumerate(frames):
        det = detect_full_frame(frame, rotation=rotation, params=p)
        shape = tuple(det.get("display_shape") or (0, 0))
        H, W = shape
        issues = []
        warnings = []
        if i < len(rects) and rects[i] is not None:
            rect = tuple(int(v) for v in rects[i])
        else:
            rect = det.get("even_rect") or (0, 0, 0, 0)
            issues.append("missing rectangle, using detection")

        x, y, w, h = rect
        if w % 2 or h % 2:
            issues.append("odd crop dimensions")
        if x < 0 or y < 0 or x + w > W or y + h > H:
            issues.append("crop rectangle outside display frame")

        # A bar present in this frame that the rectangle fails to remove is a
        # real defect: black is exposed on that side.
        bars = det.get("bars") or {}
        if bars.get("top", 0) > y:
            issues.append("residual top bar not cropped")
        if bars.get("bottom", 0) > H - (y + h):
            issues.append("residual bottom bar not cropped")
        if bars.get("left", 0) > x:
            issues.append("residual left bar not cropped")
        if bars.get("right", 0) > W - (x + w):
            issues.append("residual right bar not cropped")

        # The other direction: remove picture that this frame calls content.
        # Cutting content means the rectangle starts inside the content
        # rectangle, i.e. x > cx / y > cy or ends before it.
        content = det.get("rect") or (0, 0, W, H)
        cx, cy, cw, ch = content
        if cw > 0 and ch > 0:
            cut = max(0, x - cx) + max(0, y - cy) + \
                max(0, (cx + cw) - (x + w)) + max(0, (cy + ch) - (y + h))
            if cut > 0:
                warnings.append(f"crop removes {cut} content pixels on this frame")

        ok = not issues
        all_ok = all_ok and ok
        items.append({
            "index": i,
            "ok": ok,
            "issues": issues,
            "warnings": warnings,
            "rect": rect,
            "detected_rect": det.get("even_rect"),
            "bars": bars,
            "caption_protected": list(det.get("protected_sides") or ()),
            "display_shape": shape,
            "rect_outside": bool(
                x < 0 or y < 0 or x + w > W or y + h > H
            ),
        })

    return {
        "ok": all_ok,
        "frames": len(items),
        "items": items,
        "issues": [issue for item in items for issue in item["issues"]],
        "warnings": [w for item in items for w in item["warnings"]],
    }


def review_result(cropped, source=None, rects=None, rotation=0, params=None):
    """Verify cropped output against the source and the chosen rectangles.

    Checks: cropped dimensions are even, they match the rectangles, the
    rectangles are inside the display frame, they match full-frame detection,
    and no bar was left outside the crop.  Also reports any side whose crop was
    skipped for caption protection, which is the desired safe behaviour.
    """

    cropped = _frame_list(cropped)
    issues = []
    items = []

    if not cropped:
        return {"ok": False, "issues": ["no cropped frames"], "frames": 0}

    rects = list(rects or [])
    audits = audit(source, rects, rotation=rotation, params=params) if source is not None else None

    for i, frame in enumerate(cropped):
        shape = tuple(int(v) for v in frame.shape[:2])
        item_issues = []
        if shape[0] % 2 or shape[1] % 2:
            item_issues.append("odd cropped dimensions")
        rect = None
        if i < len(rects) and rects[i] is not None:
            rect = tuple(int(v) for v in rects[i])
            if shape != (rect[3], rect[2]):
                item_issues.append("cropped size does not match rectangle")

        # Look back at the output itself: re-run detection on the cropped
        # frame.  Bars must be gone, except on sides that were deliberately
        # kept to protect a caption.
        residual = detect_full_frame(frame, rotation=0, params=params)
        residual_bars = residual["bars"]
        p = _params(params)
        allowed_sides = set()
        if audits is not None and rect is not None:
            allowed_sides = set(audits["items"][i]["caption_protected"])
            if not audits["items"][i]["ok"]:
                item_issues.extend(audits["items"][i]["issues"])
        survivors = [
            side for side in ("top", "bottom", "left", "right")
            if residual_bars.get(side, 0) > p["residual_tol"]
            and side not in allowed_sides
        ]
        if survivors:
            item_issues.append(
                "residual black bar after crop: " + ",".join(survivors)
            )
        items.append({
            "index": i,
            "ok": not item_issues,
            "issues": item_issues,
            "shape": shape,
            "rect": rect,
            "residual_bars": dict(residual_bars),
            "allowed_residual_sides": sorted(allowed_sides),
        })
        issues.extend(item_issues)

    ok = not issues
    protected = 0
    if audits is not None:
        protected = sum(1 for it in audits["items"] if it["caption_protected"])
    return {
        "ok": ok,
        "issues": issues,
        "frames": len(items),
        "items": items,
        "caption_protected_frames": protected,
        "audit": audits,
        "warnings": list((audits or {}).get("warnings") or []),
    }


# ---------------------------------------------------------------------------


def fallback(frame, detection=None, params=None):
    """Conservative result when detection is invalid or suspect.

    Returning the full frame (no crop) can never cut away picture or captions;
    the caller can still read ``rect`` and the diagnostic flags.
    """

    p = _params(params)
    result = {
        "rect": (0, 0, 0, 0),
        "even_rect": (0, 0, 0, 0),
        "used": True,
        "reason": "invalid frame",
        "bars": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        "flags": {side: "fallback" for side in ("top", "bottom", "left", "right")},
        "display_shape": (0, 0),
        "ok": False,
        "params": p,
    }
    if not _is_frame(frame):
        return result

    shape = tuple(int(v) for v in frame.shape[:2])
    det = detection if isinstance(detection, dict) else None
    if det and det.get("ok") and not det.get("suspect"):
        result.update(
            rect=det.get("even_rect") or det.get("rect") or (0, 0, shape[1], shape[0]),
            even_rect=det.get("even_rect") or (0, 0, shape[1], shape[0]),
            bars=dict(det.get("bars") or {}),
            flags=dict(det.get("flags") or {}),
            display_shape=det.get("display_shape") or shape,
            used=False,
            reason="detection valid",
            ok=True,
        )
        return result

    rect, _ = _align_even_rect((0, 0, shape[1], shape[0]), shape)
    result.update(
        rect=rect,
        even_rect=rect,
        display_shape=det.get("display_shape") if det else shape,
        reason="detection absent or suspect; keeping the full frame",
        ok=True,
    )
    return result


def determinism(source, rotation=0, params=None, runs=2):
    """Run the detection/tracking pipeline repeatedly and compare results."""

    results = []
    for _ in range(max(1, int(runs))):
        if _is_frame(source):
            results.append(detect_full_frame(source, rotation=rotation, params=params))
        else:
            results.append(track_per_frame(source, rotation=rotation, params=params))

    keys = []
    for result in results:
        rects = result.get("rects") if "rects" in result else [result.get("even_rect")]
        bars = result.get("bars") if "bars" in result else [result.get("bars")]
        keys.append((tuple(rects), tuple(str(b) for b in bars)))
    same = all(key == keys[0] for key in keys[1:]) if keys else True
    return {
        "ok": bool(same),
        "same": bool(same),
        "runs": len(results),
        "rects": results[0].get("rects") if "rects" in results[0]
        else [results[0].get("even_rect")],
    }


# ---------------------------------------------------------------------------
# I/O: load, export
# ---------------------------------------------------------------------------

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def load(source, rotation=None, max_frames=None):
    """Load frames and their rotation metadata.

    Returns a bundle ``{"frames", "rotation", "fps", "storage_shape",
    "display_shape", "path", "ok"}``.  Arrays/lists are passed through.  Files
    are read with auto-orientation disabled so that the raw storage frames and
    the rotation metadata are kept separate; rotation happens later, in
    display space.
    """

    bundle = {
        "frames": [],
        "rotation": 0,
        "fps": 0.0,
        "storage_shape": (0, 0),
        "display_shape": (0, 0),
        "path": None,
        "ok": False,
        "reason": "unsupported source",
    }

    if isinstance(source, dict) and "frames" in source:
        frames = _frame_list(source.get("frames"))
        rot = source.get("rotation", 0) if rotation is None else rotation
        rot = _normalise_rotation(rot)
        shapes = [tuple(int(v) for v in f.shape[:2]) for f in frames]
        bundle.update(
            frames=frames,
            rotation=rot,
            fps=float(source.get("fps", 0.0) or 0.0),
            storage_shape=shapes[0] if shapes else (0, 0),
            display_shape=_display_shape(shapes[0], rot) if shapes else (0, 0),
            path=source.get("path"),
            ok=bool(frames),
            reason="bundle",
        )
        return bundle

    if _is_frame(source) or (
        isinstance(source, (list, tuple))
        and source
        and _coerce_frame(source[0]) is not None
    ):
        frames = _frame_list(source)
        rot = _normalise_rotation(0 if rotation is None else rotation)
        shapes = [tuple(int(v) for v in f.shape[:2]) for f in frames]
        bundle.update(
            frames=frames,
            rotation=rot,
            fps=0.0,
            storage_shape=shapes[0] if shapes else (0, 0),
            display_shape=_display_shape(shapes[0], rot) if shapes else (0, 0),
            ok=bool(frames),
            reason="frames",
        )
        return bundle

    if not isinstance(source, (str, Path)):
        return bundle

    path = Path(source)
    if not path.exists():
        bundle["reason"] = "missing file"
        return bundle

    bundle["path"] = str(path)
    suffix = path.suffix.lower()

    if suffix in _IMAGE_SUFFIXES:
        if cv2 is None:
            bundle["reason"] = "opencv unavailable"
            return bundle
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            bundle["reason"] = "unreadable image"
            return bundle
        bundle.update(
            frames=[img],
            rotation=0,
            fps=0.0,
            storage_shape=tuple(int(v) for v in img.shape[:2]),
            display_shape=tuple(int(v) for v in img.shape[:2]),
            ok=True,
            reason="image",
        )
        return bundle

    if cv2 is None:
        bundle["reason"] = "opencv unavailable"
        return bundle

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        bundle["reason"] = "unreadable video"
        return bundle

    try:
        auto = cv2.CAP_PROP_ORIENTATION_AUTO
        cap.set(auto, 0)  # keep raw storage frames; rotate later ourselves
    except Exception:
        pass

    rot = 0
    try:
        raw_rot = float(cap.get(cv2.CAP_PROP_ORIENTATION_META))
        if math.isfinite(raw_rot):
            rot = _normalise_rotation(raw_rot)
    except Exception:
        rot = 0
    if rotation is not None:
        rot = _normalise_rotation(rotation)

    fps = 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    except Exception:
        fps = 0.0

    frames = []
    limit = int(max_frames) if max_frames else None
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
        if limit and len(frames) >= limit:
            break
    cap.release()

    if not frames:
        bundle["reason"] = "no frames decoded"
        return bundle

    storage = tuple(int(v) for v in frames[0].shape[:2])
    bundle.update(
        frames=frames,
        rotation=rot,
        fps=fps,
        storage_shape=storage,
        display_shape=_display_shape(storage, rot),
        ok=True,
        reason="video",
    )
    return bundle


def _even_crop(crop_frame):
    """Trim right/bottom by one pixel to make a frame even (no padding)."""

    if not _is_frame(crop_frame):
        return crop_frame, False
    h, w = crop_frame.shape[:2]
    trimmed = (h % 2) or (w % 2)
    if h % 2:
        h -= 1
    if w % 2:
        w -= 1
    return np.ascontiguousarray(crop_frame[:h, :w]), bool(trimmed)


def export(result, path=None, fps=24.0, codec="mp4v", even=True):
    """Export the processed frames, guaranteeing even dimensions.

    ``result`` may be a crop bundle (``{"frames", "rects"}``), a list of frames,
    or a single frame.  When ``path`` is ``None`` nothing is written and the
    validated metadata is returned.  Odd frames are trimmed (never padded), so
    an encoder cannot re-introduce a black line.

    ``export(path, frames, ...)`` argument order is also accepted.
    """

    # Tolerate the reversed export(path, frames, ...) calling convention.
    if isinstance(result, (str, Path)) and not isinstance(path, (str, Path)):
        result, path = path, result

    frames = None
    rects = None
    rotation = 0
    if isinstance(result, dict):
        frames = _frame_list(result.get("frames", []))
        rects = result.get("rects")
        rotation = result.get("rotation", 0)
        fps = float(result.get("fps", fps) or fps or 24.0)
    else:
        frames = _frame_list(result)

    info = {
        "ok": False,
        "path": str(path) if path is not None else None,
        "frames": 0,
        "shape": (0, 0),
        "even": True,
        "trimmed": 0,
        "issues": [],
        "rotation": _normalise_rotation(rotation),
    }
    if not frames:
        info["issues"].append("no frames to export")
        return info

    prepared = []
    trimmed_count = 0
    for frame in frames:
        if even:
            frame, trimmed = _even_crop(frame)
            trimmed_count += int(trimmed)
        prepared.append(frame)
    shapes = [tuple(int(v) for v in f.shape[:2]) for f in prepared]
    if any(shape[0] % 2 or shape[1] % 2 for shape in shapes):
        info["even"] = False
        info["issues"].append("odd frame dimensions after trimming")
    if len({shape for shape in shapes}) > 1:
        info["issues"].append("inconsistent frame sizes")
    info["trimmed"] = trimmed_count
    info["frames"] = len(prepared)
    info["shape"] = shapes[0]

    if path is None:
        info["ok"] = info["even"] and not info["issues"]
        return info

    if cv2 is None:
        info["issues"].append("opencv unavailable")
        return info

    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_SUFFIXES:
        cv2.imwrite(path, prepared[0])
        info["ok"] = True
        return info

    writer = None
    try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, float(fps or 24.0), (shapes[0][1], shapes[0][0]))
        if not writer.isOpened():
            writer.release()
            writer = None
            info["issues"].append("could not open video writer")
            return info
        for frame in prepared:
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()

    info["ok"] = Path(path).exists() and not info["issues"]
    return info


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------

def process(source, rotation=None, params=None, path=None, fps=None):
    """Load -> detect/track -> crop -> review -> (optional) export.

    The crop rectangles are computed in display space on the full frame, tracked
    per frame and snapped to even dimensions; the review pass then re-checks the
    result against a fresh full-frame detection.
    """

    bundle = load(source, rotation=rotation, max_frames=None)
    if not bundle["ok"]:
        return {"ok": False, "issues": [bundle.get("reason")], "bundle": bundle,
                "tracking": None, "cropped": [], "review": None, "export": None}

    frames = bundle["frames"]
    rot = bundle["rotation"]
    tracking = track_per_frame(frames, rotation=rot, params=params)
    cropped = [
        crop(frame, rect=rect, rotation=rot, rect_space="display")
        for frame, rect in zip(frames, tracking["rects"])
    ]
    review = review_result(cropped, source=frames, rects=tracking["rects"],
                           rotation=rot, params=params)
    if path is None:
        export_info = export(cropped)
    else:
        export_info = export(
            {"frames": cropped, "rects": tracking["rects"], "rotation": rot,
             "fps": fps or bundle.get("fps") or 24.0},
            path=path,
        )
    ok = bool(tracking["ok"] and review["ok"] and export_info["ok"])
    return {
        "ok": ok,
        "rotation": rot,
        "fps": fps or bundle.get("fps") or 24.0,
        "tracking": tracking,
        "cropped": cropped,
        "review": review,
        "export": export_info,
        "bundle": bundle,
        "issues": list(review.get("issues") or []),
    }

