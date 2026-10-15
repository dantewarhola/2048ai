import argparse
import ctypes
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageGrab

SIZE = 4

BOARD_COLOR = (187, 173, 160)  # #bbada0, the grid background / gaps

EMPTY_COLOR = (205, 193, 180)  # #cdc1b4, an empty cell

# Tile values are read from the number printed on each tile, not its color
# (on 2048.org every tile from 4096 up is the same dark color). Each tile's
# text is compared with every possible value, drawn in a bold system font.
VALUES = [2 ** k for k in range(1, 18)]  # 2 .. 131072
TEXT_FONT = "arialbd.ttf"  # Arial Bold, close enough to the game's Clear Sans Bold
TEXT_HEIGHT = 24           # text is compared at this height in pixels
TEXT_CONTRAST = 60         # min color distance between text and tile background
CELL_PIXELS = 64           # cells are shrunk to about this size before reading


def _render(text, font):
    """White-on-black image of `text`, cropped to the ink."""
    img = Image.new("L", (900, 200), 0)
    ImageDraw.Draw(img).text((10, 10), text, fill=255, font=font)
    ys, xs = np.nonzero(np.asarray(img) > 128)
    return img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


_TEMPLATES = None


def templates():
    """(value, aspect ratio, image) for every possible tile value, made on first use."""
    global _TEMPLATES
    if _TEMPLATES is None:
        font = ImageFont.truetype(TEXT_FONT, 120)
        _TEMPLATES = []
        for v in VALUES:
            img = _render(str(v), font)
            _TEMPLATES.append((v, img.width / img.height, img))
    return _TEMPLATES


# Tiles of 128 and up glow: the game draws a gold shadow around them, which
# tints the nearby board border, gaps and empty cells. A tinted pixel is
# (1 - a) * color + a * GLOW_COLOR for some strength a.
GLOW_COLOR = (243, 215, 116)
MAX_GLOW = 0.7  # strongest tint expected (overlapping 2048 glows); a 128 tile is ~0.9


def glow_distance(pixels, color, max_glow=MAX_GLOW):
    """Distance from each pixel (..., 3) to `color` with any glow tint up to max_glow."""
    v = np.asarray(pixels, dtype=float) - color
    d = np.subtract(GLOW_COLOR, color, dtype=float)
    a = np.clip(v @ d / (d @ d), 0.0, max_glow)
    return np.linalg.norm(v - a[..., None] * d, axis=-1)


def board_mask(arr, tol=9):
    """True where a pixel is the board's background color, glow-tinted or not."""
    return glow_distance(arr, BOARD_COLOR) <= tol


# Fraction of the board width taken by one gap (15px of a 500px board in the
# original CSS; cells are 106.25px).
GAP_RATIO = 15 / 500


def capture_screen():
    # Make the process DPI-aware so the capture uses real pixels on scaled displays.
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    return ImageGrab.grab(all_screens=True)


def find_board(img):
    """Locate the board's bounding box (left, top, right, bottom) in the image."""
    arr = np.asarray(img.convert("RGB"), dtype=int)
    mask = board_mask(arr)
    if mask.sum() < 1000:
        return None

    # The board's border and gaps form unbroken horizontal lines of board color
    # spanning its full width. Find the longest such line in each row; the
    # board's is the longest on screen (other UI, like the score boxes, may
    # share the color but is narrower).
    runs = []  # (length, start) of the longest run per row
    for row in mask:
        padded = np.concatenate(([0], row.astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(padded))
        if len(edges) == 0:
            runs.append((0, 0))
            continue
        starts, ends = edges[::2], edges[1::2]
        i = int((ends - starts).argmax())
        runs.append((int(ends[i] - starts[i]), int(starts[i])))

    best_len = max(r[0] for r in runs)
    if best_len < 100:
        return None
    # Rows whose line matches the board's full width (allowing a few px of
    # anti-aliasing at the rounded corners).
    left = min(s for l, s in runs if l >= best_len - 4)
    full_rows = [y for y, (l, s) in enumerate(runs)
                 if l >= best_len * 0.95 and abs(s - left) <= best_len * 0.05]
    top, bottom = full_rows[0], full_rows[-1] + 1
    right = left + best_len

    if bottom - top < best_len * 0.8:
        return None
    return left, top, right, bottom


_RESIZED = {}


def _resized(value, img, size):
    """Template image at `size`, as a 0..1 array (cached: sizes repeat every frame)."""
    key = (value, size)
    if key not in _RESIZED:
        _RESIZED[key] = np.asarray(img.resize(size, Image.BILINEAR), dtype=float) / 255
    return _RESIZED[key]


def read_number(ink):
    """The tile value whose drawn number best matches `ink` (2D, 0..1, cropped to the text)."""
    h, w = ink.shape
    aspect = w / h
    size = (max(4, round(TEXT_HEIGHT * aspect)), TEXT_HEIGHT)
    seen = np.asarray(Image.fromarray((ink * 255).astype(np.uint8)).resize(size, Image.BILINEAR),
                      dtype=float) / 255
    best, best_score = None, np.inf
    for value, t_aspect, img in templates():
        shape_off = abs(np.log(aspect / t_aspect))  # mostly the number of digits
        if shape_off > 0.5:
            continue
        drawn = _resized(value, img, size)
        score = np.abs(seen - drawn).mean() + 0.5 * shape_off
        if score < best_score:
            best, best_score = value, score
    return best


def read_cell(patch):
    """Value of one cell from its pixels: 0 if empty, None if unreadable."""
    step = max(1, patch.shape[0] // CELL_PIXELS)
    p = patch[::step, ::step].astype(float)
    h, w, _ = p.shape
    m = max(2, int(h * 0.06))
    # The cell's edges never have text on them, so they give its background.
    edges = np.concatenate([p[:m].reshape(-1, 3), p[-m:].reshape(-1, 3),
                            p[:, :m].reshape(-1, 3), p[:, -m:].reshape(-1, 3)])
    background = np.median(edges, axis=0)
    dist = np.linalg.norm(p - background, axis=2)
    dist[:m] = dist[-m:] = 0
    dist[:, :m] = dist[:, -m:] = 0
    text = dist > TEXT_CONTRAST
    if text.sum() < h * w * 0.005:
        # No number: an empty cell (maybe tinted by a neighbor's glow), or a
        # tile caught mid-animation.
        return 0 if glow_distance(background, EMPTY_COLOR) <= 25 else None
    ys, xs = np.nonzero(text)
    ink = dist[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return read_number(np.clip(ink / np.percentile(dist[text], 90), 0, 1))


def board_at(img, box):
    """True if the board is still at `box`: its top and left borders are board color."""
    left, top, right, bottom = box
    edge = max(1, int((right - left) * GAP_RATIO / 2))  # middle of the border strip
    top_strip = img.crop((left + edge, top + edge, right - edge, top + edge + 1))
    left_strip = img.crop((left + edge, top + edge, left + edge + 1, bottom - edge))
    for strip in (top_strip, left_strip):
        arr = np.asarray(strip.convert("RGB"), dtype=int).reshape(-1, 3)
        if not board_mask(arr).all():
            return False
    return True


def read_board(img, box=None):
    """Read the grid; returns (grid, box), or (None, None) if no board is found.

    Pass the box from the previous read to skip searching the whole image
    while the board hasn't moved.
    """
    if box is None or not board_at(img, box):
        box = find_board(img)
        if box is None:
            return None, None
    left, top, right, bottom = box
    arr = np.asarray(img.crop(box).convert("RGB"), dtype=int)  # board only

    w, h = right - left, bottom - top
    gap_x, gap_y = w * GAP_RATIO, h * GAP_RATIO
    cell_w, cell_h = (w - gap_x * (SIZE + 1)) / SIZE, (h - gap_y * (SIZE + 1)) / SIZE

    grid = []
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            x0, y0 = int(gap_x * (c + 1) + cell_w * c), int(gap_y * (r + 1) + cell_h * r)
            row.append(read_cell(arr[y0:y0 + int(cell_h), x0:x0 + int(cell_w)]))
        grid.append(row)
    return grid, box


def format_board(grid):
    def cell(v):
        if v is None:
            return "?"
        if v == 0:
            return "."
        return str(v)

    width = max(5, max(len(cell(v)) for row in grid for v in row))
    line = "+" + "+".join("-" * (width + 2) for _ in range(SIZE)) + "+"
    out = [line]
    for row in grid:
        out.append("|" + "|".join(f" {cell(v):>{width}} " for v in row) + "|")
        out.append(line)
    return "\n".join(out)


VK_Q = 0x51

# Set by the desktop app (app.py): the path of a file whose appearance means
# "stop", so the Stop button works like pressing Q.
STOP_FILE_ENV = "AI2048_STOP_FILE"


def stop_requested():
    path = os.environ.get(STOP_FILE_ENV)
    return bool(path) and os.path.exists(path)


def q_pressed():
    """True if Q was pressed (in any window) or the app asked us to stop."""
    return stop_requested() or bool(ctypes.windll.user32.GetAsyncKeyState(VK_Q) & 0x8001)


def wait(seconds):
    """Sleep for up to `seconds`, returning True early if Q is pressed."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if q_pressed():
            return True
        time.sleep(0.02)
    return q_pressed()


def main():
    parser = argparse.ArgumentParser(description="Print the 2048 board shown on screen.")
    parser.add_argument("--image", help="read from an image file instead of the screen")
    parser.add_argument("--interval", type=float, default=0.02, help="seconds between screen reads")
    args = parser.parse_args()

    if args.image:
        grid, _ = read_board(Image.open(args.image))
        if grid is None:
            sys.exit("No 2048 board found in image.")
        print(format_board(grid))
        return

    print("Watching the screen for a 2048 board. Press Q (in any window) to quit.", flush=True)
    q_pressed()  # clear any stale "pressed since last check" state
    last = None       # last board printed
    previous = None   # board from the previous read
    searching = False
    box = None        # where the board was last seen
    while True:
        grid, box = read_board(capture_screen(), box)
        if grid is None:
            if not searching:
                print("No 2048 board visible yet, still looking...", flush=True)
                searching = True
                last = None  # reprint the board when it reappears, even if unchanged
        else:
            searching = False
            # Only print a board once it has read identically twice in a row.
            # Mid-animation frames (tiles slid but the new tile not yet spawned,
            # or a tile still fading in) change between reads, so they're skipped.
            # Boards with unreadable cells are always mid-animation.
            settled = grid == previous and all(v is not None for row in grid for v in row)
            if settled and grid != last:
                print(format_board(grid), flush=True)
                print(flush=True)
                last = grid
        previous = grid
        if wait(args.interval):
            print("Q pressed, exiting.")
            return


if __name__ == "__main__":
    main()
