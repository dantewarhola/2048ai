"""N-tuple network agent for 2048, trained with TD(0) learning on afterstates.

How it learns
-------------
On every turn of every game the agent tries all 4 possible moves. For each one
it computes the afterstate (the board after sliding, before the random tile
spawns) and scores it as

    points earned by the move + V(afterstate)

where V is the network's estimate of all future points from that afterstate.
It plays the best-scoring move. Once the next turn's best move is known, the
previous move is corrected toward what actually happened:

    V(previous afterstate) += alpha * (points + V(next afterstate) - V(previous afterstate))

and toward 0 when the game ended. So the agent learns from every move it
makes, using its judgement of all the moves it could have made.

The network
-----------
V(board) is the sum of weights looked up from 4 tables, one per "6-tuple" (a
fixed group of 6 cells). Each tuple is read in all 8 rotations/reflections
of the board, so one board gives 32 lookups. Each cell holds a tile exponent
0-15 (tiles above 32768 are clipped to 32768), so a table has 16^6 entries
and the network is 4 * 16^6 float32 weights, about 270 MB.

Many games run at once (see game2048.Game2048). Their updates are combined
per weight by averaging, so a large batch doesn't overshoot on common boards.

Usage
-----
    python agent2048.py train                    # train (Ctrl+C saves and stops)
    python agent2048.py train --games 500000     # train for a set number of games
    python agent2048.py eval --games 1000        # measure win rates with expectimax depth 1
    python agent2048.py eval --depth 0           # greedy play (no search), much faster
    python agent2048.py train --corner 0         # train without the corner strategy
Training resumes from agent_weights.npy if it exists.

Corner strategy
---------------
By default moves are picked by points + V(afterstate) + corner_weight * a
"snake" score that rewards keeping the biggest tile in the bottom-right corner
with the next biggest lined up beside it (see corner_scores). The network
itself is symmetric, so it can't prefer a corner; the snake score does that,
while training with it teaches V the value of the corner-style positions the
agent actually plays into.
"""

import argparse
import json
import math
import os
import shutil
import sys
import time
from collections import deque

import numpy as np

from game2048 import N_ACTIONS, SIZE, Game2048, slide
from runtime import DATA_DIR

DEFAULT_WEIGHTS = os.path.join(DATA_DIR, "agent_weights.npy")

# Set by the desktop app (app.py): the path of a file whose appearance means
# "stop", so its Stop button works like Ctrl+C (training saves first).
STOP_FILE_ENV = "AI2048_STOP_FILE"


def stop_requested():
    path = os.environ.get(STOP_FILE_ENV)
    return bool(path) and os.path.exists(path)

# Cell numbers: row * 4 + column.
TUPLES = [
    (0, 1, 2, 3, 4, 5),     # top row + 2 cells below
    (4, 5, 6, 7, 8, 9),     # second row + 2 cells below
    (0, 1, 2, 4, 5, 6),     # 2x3 block in the corner
    (4, 5, 6, 8, 9, 10),    # 2x3 block one row down
]
N_VALUES = 16                     # exponents 0..15 per cell
TUPLE_LEN = 6
TABLE_SIZE = N_VALUES ** TUPLE_LEN


def _symmetries():
    """The 8 rotations/reflections of the board, as cell-index maps."""
    cells = np.arange(SIZE * SIZE).reshape(SIZE, SIZE)
    maps = []
    for grid in (cells, cells[:, ::-1]):
        for k in range(4):
            maps.append(np.rot90(grid, k).ravel())
    return maps


# FEATURE_CELLS[f] = the 6 board cells read by feature f (tuple x symmetry).
FEATURE_CELLS = np.array([m[list(t)] for t in TUPLES for m in _symmetries()])
FEATURE_OFFSET = np.repeat(np.arange(len(TUPLES)) * TABLE_SIZE, 8)
N_FEATURES = len(FEATURE_CELLS)  # 32
_DIGITS = N_VALUES ** np.arange(TUPLE_LEN - 1, -1, -1)


# Snake path starting in the bottom-right corner: along the bottom row to the
# left, back along the row above to the right, and so on. SNAKE_RANK[r, c] is
# the cell's position on that path (0 = bottom-right corner).
SNAKE_RANK = np.array([
    [12, 13, 14, 15],
    [11, 10, 9, 8],
    [4, 5, 6, 7],
    [3, 2, 1, 0],
])
DEFAULT_CORNER = 128.0  # snake heuristic weight used for training and play
SNAKE_PATH = np.argsort(SNAKE_RANK.ravel())  # cell numbers in path order
# Each step along the path halves a cell's weight, so a board scores highest
# with its biggest tile in the corner and the next biggest following the path.
SNAKE_WEIGHTS = 0.5 ** np.arange(SIZE * SIZE)  # by path position


def corner_scores(boards):
    """(n, 4, 4) exponent boards -> how well each follows the bottom-right snake.

    The sum of tile values weighted by their position on the snake path, minus
    a penalty wherever a tile is bigger than the one before it on the path
    (that breaks the chain of merges leading back to the corner).
    """
    flat = boards.reshape(len(boards), SIZE * SIZE)[:, SNAKE_PATH].astype(np.int64)
    path = np.where(flat > 0, 2.0 ** flat, 0.0)
    rises = np.maximum(path[:, 1:] - path[:, :-1], 0.0)
    return path @ SNAKE_WEIGHTS - 2 * rises @ SNAKE_WEIGHTS[1:]


def features(boards):
    """(n, 4, 4) exponent boards -> (n, 32) indices into the weight array."""
    flat = np.minimum(boards.reshape(len(boards), -1), N_VALUES - 1).astype(np.int64)
    return (flat[:, FEATURE_CELLS] * _DIGITS).sum(axis=2) + FEATURE_OFFSET


class Agent:
    def __init__(self, weights=None, corner_weight=0.0):
        if weights is None:
            weights = np.zeros(len(TUPLES) * TABLE_SIZE, dtype=np.float32)
        self.w = weights
        # > 0 adds corner_weight * corner_scores() to every afterstate's value,
        # steering play toward the bottom-right snake. Training leaves it at 0.
        self.corner_weight = corner_weight

    # ----- persistence

    @classmethod
    def load(cls, path=DEFAULT_WEIGHTS, corner_weight=0.0):
        if path.endswith(".npz"):
            return cls(load_compact(path)[0], corner_weight)
        return cls(np.load(path), corner_weight)

    def save(self, path=DEFAULT_WEIGHTS, info=None):
        tmp = path + ".tmp.npy"
        np.save(tmp, self.w)
        os.replace(tmp, path)
        if info is not None:
            with open(os.path.splitext(path)[0] + ".json", "w") as f:
                json.dump(info, f, indent=2)

    # ----- evaluation

    def value(self, feats):
        return self.w[feats].sum(axis=1, dtype=np.float64)

    def evaluate_moves(self, boards):
        """Try all 4 moves on each board.

        Returns (actions, points, values, feats, can_move): the best move per
        board, the points and afterstate value of that move, the afterstate's
        features, and whether any move was possible at all.

        The corner heuristic only steers which move is picked; `values` is the
        network's own estimate, so training keeps V an estimate of real points.
        """
        n = len(boards)
        after = np.empty((N_ACTIONS, n, SIZE, SIZE), dtype=boards.dtype)
        points = np.empty((N_ACTIONS, n))
        valid = np.empty((N_ACTIONS, n), dtype=bool)
        for a in range(N_ACTIONS):
            after[a], points[a], valid[a] = slide(boards, a)

        feats = features(after.reshape(-1, SIZE, SIZE)).reshape(N_ACTIONS, n, N_FEATURES)
        values = self.value(feats.reshape(-1, N_FEATURES)).reshape(N_ACTIONS, n)
        total = points + values
        if self.corner_weight:
            total += self.corner_weight * corner_scores(after.reshape(-1, SIZE, SIZE)).reshape(N_ACTIONS, n)
        total = np.where(valid, total, -np.inf)

        best = total.argmax(axis=0)
        cols = np.arange(n)
        return best, points[best, cols], values[best, cols], feats[best, cols], valid.any(axis=0)

    def best_move(self, board, depth=0):
        """Best action (0 up, 1 left, 2 down, 3 right) for one (4, 4) exponent board,
        or None if no move is possible. depth > 0 uses expectimax search."""
        boards = np.asarray(board, dtype=np.int8)[None]
        action, can_move = self.search_moves(boards, depth)
        return int(action[0]) if can_move[0] else None

    # ----- expectimax search

    def board_values(self, boards, chunk=20000):
        """V for many afterstates, in chunks to bound memory."""
        out = np.empty(len(boards))
        for i in range(0, len(boards), chunk):
            out[i:i + chunk] = self.value(features(boards[i:i + chunk]))
        if self.corner_weight:
            out += self.corner_weight * corner_scores(boards)
        return out

    def _expected(self, afters, depth):
        """Expected future points from each afterstate, searching `depth` more
        rounds of (random tile spawn -> best move) before trusting V."""
        if depth == 0 or len(afters) == 0:
            return self.board_values(afters)

        m = len(afters)
        flat = afters.reshape(m, -1)
        parent, cell = np.nonzero(flat == 0)
        n_empty = np.bincount(parent, minlength=m)

        # Every possible spawn: each empty cell gets a 2 (90%) or a 4 (10%),
        # with the cell chosen uniformly.
        children = np.repeat(flat[parent], 2, axis=0)
        rows = np.arange(len(children))
        children[rows, np.repeat(cell, 2)] = np.tile([1, 2], len(parent))
        prob = np.repeat(1.0 / n_empty[parent], 2) * np.tile([0.9, 0.1], len(parent))
        child_parent = np.repeat(parent, 2)
        children = children.reshape(-1, SIZE, SIZE)

        # After each spawn, the player picks the best move (or the game ends: 0).
        best = np.full(len(children), -np.inf)
        for a in range(N_ACTIONS):
            after, points, valid = slide(children, a)
            q = np.full(len(children), -np.inf)
            q[valid] = points[valid] + self._expected(after[valid], depth - 1)
            best = np.maximum(best, q)
        best[np.isneginf(best)] = 0.0

        result = np.bincount(child_parent, weights=prob * best, minlength=m)
        # An afterstate always has an empty cell, but fall back to V just in case.
        no_spawn = n_empty == 0
        if no_spawn.any():
            result[no_spawn] = self.board_values(afters[no_spawn])
        return result

    def search_moves(self, boards, depth=1):
        """Best move for each board using expectimax `depth` levels deep
        (0 = greedy on V alone). Returns (actions, can_move)."""
        if depth == 0:
            actions, _, _, _, can_move = self.evaluate_moves(boards)
            return actions, can_move

        # Search a few boards at a time; the tree grows ~100x per level.
        chunk = max(1, 256 // 100 ** (depth - 1))
        actions = np.zeros(len(boards), dtype=np.int64)
        can_move = np.zeros(len(boards), dtype=bool)
        for i in range(0, len(boards), chunk):
            part = boards[i:i + chunk]
            q = np.full((N_ACTIONS, len(part)), -np.inf)
            for a in range(N_ACTIONS):
                after, points, valid = slide(part, a)
                q[a, valid] = points[valid] + self._expected(after[valid], depth)
            actions[i:i + chunk] = q.argmax(axis=0)
            can_move[i:i + chunk] = np.isfinite(q).any(axis=0)
        return actions, can_move

    # ----- learning

    def update(self, feats, targets, alpha):
        """Move V(afterstate) toward each target; one row of feats per afterstate."""
        if len(feats) == 0:
            return
        err = targets - self.value(feats)
        step = np.repeat(err * (alpha / N_FEATURES), N_FEATURES)
        idx, inverse = np.unique(feats.ravel(), return_inverse=True)
        # Average the batch's updates per weight so many games touching the same
        # weight in one step don't add up to a huge jump.
        mean_step = np.bincount(inverse, weights=step) / np.bincount(inverse)
        self.w[idx] += mean_step.astype(np.float32)


# ---------------------------------------------------------------- compact files

def save_compact(weights, path, info=None):
    """Save weights as a compressed .npz of only the non-zero entries.

    Most of the table is never visited in training, so this is several times
    smaller than the .npy file (small enough for a GitHub release download).
    """
    index = np.flatnonzero(weights).astype(np.int64)
    gaps = np.diff(index, prepend=0).astype(np.uint32)  # small numbers compress well
    np.savez_compressed(path, gaps=gaps, values=weights[index].astype(np.float32),
                        size=np.int64(weights.size), info=json.dumps(info or {}))


def load_compact(path):
    """Read a file from save_compact. Returns (weights, info)."""
    with np.load(path) as data:
        weights = np.zeros(int(data["size"]), dtype=np.float32)
        weights[np.cumsum(data["gaps"], dtype=np.int64)] = data["values"]
        info = json.loads(str(data["info"]))
    return weights, info


# ---------------------------------------------------------------- statistics

MILESTONES = [2048, 4096, 8192, 16384, 32768]


def wilson_interval(successes, n, z=1.96):
    """95% confidence interval for a success rate."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


def summarize(scores, max_tiles):
    scores, max_tiles = np.asarray(scores), np.asarray(max_tiles)
    parts = [f"avg score {scores.mean():9,.0f}", f"best tile {max_tiles.max():6d}"]
    for m in MILESTONES[:4]:
        parts.append(f"{m}: {100 * (max_tiles >= m).mean():5.1f}%")
    return " | ".join(parts)


# Tiles tracked by the stats file / viewer.
STAT_TILES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
MAX_HISTORY = 2000


def stats_path(weights):
    """Live training stats, written while training (read by stats_viewer.py)."""
    return os.path.splitext(weights)[0] + ".stats.json"


def eval_path(weights):
    """Results of the latest `eval` run."""
    return os.path.splitext(weights)[0] + ".eval.json"


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def write_json(path, data):
    """Write atomically so a reader never sees half a file."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:  # the viewer has the file open this instant (Windows)
            time.sleep(0.05)


def tile_table(max_tiles):
    """Per tile: how often games reached it, with a 95% confidence interval."""
    max_tiles = np.asarray(max_tiles)
    n = len(max_tiles)
    rows = []
    for t in STAT_TILES:
        k = int((max_tiles >= t).sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"tile": t, "count": k, "rate": k / n if n else 0.0, "lo": lo, "hi": hi})
    return rows


def reliable_tile(table, level=0.95):
    """Highest tile reached in >= `level` of games, by the 95% CI's lower bound."""
    best = None
    for row in table:
        if row["lo"] >= level:
            best = row["tile"]
    return best


def format_duration(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


class ProgressBar:
    """A single status line, redrawn in place, showing games simulated so far.

    With a target number of games it shows a filling bar and an estimated time
    left; without one it shows a running count. Use log() to print other lines
    without breaking the bar.
    """

    SPINNER = "|/-\\"

    def __init__(self, total=None, already_trained=0, width=30, every=0.5):
        self.total = total
        self.already = already_trained
        self.width = width
        self.every = every
        self.start = time.perf_counter()
        self.last_draw = 0.0
        self.ticks = 0
        self.shown = False

    def update(self, done, force=False):
        now = time.perf_counter()
        if not force and now - self.last_draw < self.every:
            return
        self.last_draw = now
        self.ticks += 1
        elapsed = now - self.start
        rate = done / elapsed if elapsed > 0 else 0.0

        if self.total:
            frac = min(done / self.total, 1.0)
            filled = int(frac * self.width)
            bar = "#" * filled + "-" * (self.width - filled)
            eta = format_duration((self.total - done) / rate) if rate > 0 else "?"
            line = (f"[{bar}] {100 * frac:5.1f}% {done:,}/{self.total:,} games | "
                    f"{rate:,.0f}/s | {format_duration(elapsed)} elapsed, ETA {eta}")
        else:
            spin = self.SPINNER[self.ticks % len(self.SPINNER)]
            line = (f"{spin} {done:,} games this run | {rate:,.0f}/s | "
                    f"{format_duration(elapsed)} elapsed")
        line += f" | {self.already + done:,} total"

        cols = shutil.get_terminal_size((120, 20)).columns
        line = line[:cols - 1]
        sys.stdout.write("\r" + line.ljust(cols - 1))
        sys.stdout.flush()
        self.shown = True

    def log(self, text, done):
        """Print a line above the bar, then redraw the bar."""
        self.clear()
        print(text, flush=True)
        self.update(done, force=True)

    def clear(self):
        if self.shown:
            cols = shutil.get_terminal_size((120, 20)).columns
            sys.stdout.write("\r" + " " * (cols - 1) + "\r")
            sys.stdout.flush()
            self.shown = False


# ---------------------------------------------------------------- training

def train(args):
    if os.path.exists(args.weights):
        agent = Agent.load(args.weights, args.corner)
        info_path = os.path.splitext(args.weights)[0] + ".json"
        info = json.load(open(info_path)) if os.path.exists(info_path) else {}
        print(f"Resuming from {args.weights} ({info.get('games_trained', '?')} games trained).")
    else:
        agent = Agent(corner_weight=args.corner)
        info = {}
        print("Starting with a fresh network.")
    games_trained = info.get("games_trained", 0)

    env = Game2048(args.batch, seed=args.seed)
    prev_feats = np.zeros((args.batch, N_FEATURES), dtype=np.int64)
    has_prev = np.zeros(args.batch, dtype=bool)

    recent_scores = deque(maxlen=args.window)
    recent_tiles = deque(maxlen=args.window)
    recent_moves = deque(maxlen=args.window)
    finished = 0
    moves = 0
    start = last_log = last_save = last_stats = last_history = time.perf_counter()
    last_save_wall = None

    # Carry the stats history over from earlier runs of this network.
    old = read_json(stats_path(args.weights)) or {}
    history = old.get("history", []) if games_trained else []
    best_score_ever = old.get("best_score_ever", 0) if games_trained else 0
    best_tile_ever = old.get("best_tile_ever", 0) if games_trained else 0
    rate_samples = deque()  # (time, games finished) for the recent games/s rate

    def save():
        nonlocal last_save_wall
        info.update(games_trained=games_trained, alpha=args.alpha, tuples=TUPLES,
                    corner=args.corner)
        agent.save(args.weights, info)
        last_save_wall = time.time()

    def write_stats(status):
        now = time.perf_counter()
        rate_samples.append((now, finished))
        while len(rate_samples) > 2 and now - rate_samples[0][0] > 60:
            rate_samples.popleft()
        (t0, f0) = rate_samples[0]
        recent_rate = (finished - f0) / (now - t0) if now > t0 else 0.0
        table = tile_table(recent_tiles) if recent_tiles else []
        dist = {}
        for t in recent_tiles:
            dist[str(t)] = dist.get(str(t), 0) + 1
        write_json(stats_path(args.weights), {
            "status": status,
            "updated": time.time(),
            "games_trained": games_trained,
            "games_this_run": finished,
            "target_games": args.games,
            "elapsed": now - start,
            "games_per_sec": recent_rate,
            "moves_per_sec": moves / (now - start) if now > start else 0.0,
            "batch": args.batch,
            "alpha": args.alpha,
            "window": len(recent_tiles),
            "avg_score": float(np.mean(recent_scores)) if recent_scores else 0.0,
            "max_score_recent": max(recent_scores) if recent_scores else 0,
            "best_score_ever": best_score_ever,
            "best_tile_ever": best_tile_ever,
            "avg_moves": float(np.mean(recent_moves)) if recent_moves else 0.0,
            "tiles": table,
            "reliable_tile": reliable_tile(table) if table else None,
            "tile_distribution": dist,
            "last_save": last_save_wall,
            "history": history,
        })

    print(f"Training {args.batch} games at a time, alpha {args.alpha}, corner weight {args.corner}. "
          "Ctrl+C to save and stop.")
    bar = ProgressBar(args.games, already_trained=games_trained)
    bar.update(0, force=True)
    try:
        while args.games is None or finished < args.games:
            if stop_requested():
                raise KeyboardInterrupt  # stop exactly as Ctrl+C would: save, then exit
            actions, points, values, feats, can_move = agent.evaluate_moves(env.boards)

            # Learn: the previous afterstate's value should equal what followed it,
            # i.e. this move's points + value, or 0 if the game has ended.
            targets = np.where(can_move, points + values, 0.0)
            agent.update(prev_feats[has_prev], targets[has_prev], args.alpha)

            over = ~can_move
            if over.any():
                scores = env.score[over].tolist()
                tiles = env.max_tile()[over].tolist()
                recent_scores.extend(scores)
                recent_tiles.extend(tiles)
                recent_moves.extend(env.moves[over].tolist())
                best_score_ever = max(best_score_ever, max(scores))
                best_tile_ever = max(best_tile_ever, max(tiles))
                finished += int(over.sum())
                games_trained += int(over.sum())

            env.step(actions)  # games that are over are left untouched
            moves += int(can_move.sum())
            prev_feats[can_move] = feats[can_move]
            has_prev = can_move.copy()
            env.reset(over)

            now = time.perf_counter()
            if now - last_log >= args.log_every and recent_scores:
                bar.log(f"games {games_trained:9,d} | {moves / (now - start):7,.0f} moves/s | "
                        f"last {len(recent_scores)}: {summarize(recent_scores, recent_tiles)}",
                        finished)
                last_log = now
            if now - last_save >= args.save_every:
                bar.log(f"Checkpoint: saving {args.weights}...", finished)
                write_stats("saving")
                save()
                last_save = now
            # Only chart a full window: right after a (re)start, the first games to
            # finish are the short, low-scoring ones, which would show a false dip.
            if now - last_history >= args.log_every and len(recent_tiles) == recent_tiles.maxlen:
                table = tile_table(recent_tiles)
                history.append({
                    "games": games_trained,
                    "avg_score": float(np.mean(recent_scores)),
                    "rates": {str(r["tile"]): r["rate"] for r in table},
                })
                if len(history) > MAX_HISTORY:
                    history[:] = history[::2]  # thin out old points, keep the whole span
                last_history = now
            if now - last_stats >= 1.0:
                write_stats("training")
                last_stats = now
            bar.update(finished)
    except KeyboardInterrupt:
        bar.update(finished, force=True)
        print("\nStopping.")
    else:
        bar.update(finished, force=True)
        print()
    write_stats("saving")
    save()
    write_stats("stopped")
    print(f"Saved {args.weights} ({games_trained:,} games trained).")


# ---------------------------------------------------------------- evaluation

def evaluate(args):
    agent = Agent.load(args.weights, args.corner)
    env = Game2048(args.games, seed=args.seed)
    mode = f"expectimax depth {args.depth}" if args.depth else "greedy play, no search"
    print(f"Playing {args.games} games ({mode}). Ctrl+C to stop early and see results so far.")
    start = last_log = time.perf_counter()
    try:
        while not env.done.all():
            if stop_requested():
                raise KeyboardInterrupt
            active = np.flatnonzero(~env.done)
            actions = np.zeros(env.n, dtype=np.int64)
            actions[active], _ = agent.search_moves(env.boards[active], args.depth)
            env.step(actions)
            now = time.perf_counter()
            if now - last_log >= args.log_every:
                print(f"  {now - start:6.0f}s: {env.done.sum()}/{args.games} games finished, "
                      f"best tile so far {env.max_tile().max()}", flush=True)
                last_log = now
    except KeyboardInterrupt:
        print("\nStopped early; results below count finished games only.")
    elapsed = time.perf_counter() - start

    finished = env.done
    if not finished.any():
        print("No games finished.")
        return
    tiles = env.max_tile()[finished]
    scores = env.score[finished]
    n = len(tiles)
    print(f"{n} games in {elapsed:.1f}s ({mode})")
    print(f"avg score {scores.mean():,.0f}, best score {scores.max():,}, best tile {tiles.max()}")
    for m in MILESTONES:
        k = int((tiles >= m).sum())
        lo, hi = wilson_interval(k, n)
        print(f"  reached {m:5d}: {100 * k / n:5.1f}%  "
              f"(95% CI {100 * lo:5.1f}% - {100 * hi:5.1f}%)")

    info = read_json(os.path.splitext(args.weights)[0] + ".json") or {}
    table = tile_table(tiles)
    write_json(eval_path(args.weights), {
        "time": time.time(),
        "games": n,
        "depth": args.depth,
        "games_trained": info.get("games_trained"),
        "avg_score": float(scores.mean()),
        "best_score": int(scores.max()),
        "best_tile": int(tiles.max()),
        "tiles": table,
        "reliable_tile": reliable_tile(table),
        "seconds": elapsed,
    })


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate the 2048 n-tuple agent.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train the agent")
    t.add_argument("--games", type=int, default=None, help="stop after this many games (default: run until Ctrl+C)")
    t.add_argument("--batch", type=int, default=256, help="games played at once")
    t.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    t.add_argument("--window", type=int, default=1000, help="recent games shown in the log")
    t.add_argument("--log-every", type=float, default=10, help="seconds between log lines")
    t.add_argument("--save-every", type=float, default=300, help="seconds between checkpoints")
    t.add_argument("--corner", type=float, default=DEFAULT_CORNER,
                   help="weight of the bottom-right snake strategy while training (0 = off)")
    t.add_argument("--seed", type=int, default=None)
    t.add_argument("--weights", default=DEFAULT_WEIGHTS)

    e = sub.add_parser("eval", help="measure the agent without learning")
    e.add_argument("--games", type=int, default=1000)
    e.add_argument("--depth", type=int, default=1,
                   help="expectimax search depth (0 = greedy, 1 = look one spawn + move ahead, ...)")
    e.add_argument("--corner", type=float, default=DEFAULT_CORNER,
                   help="weight of the bottom-right snake heuristic (0 = network only)")
    e.add_argument("--seed", type=int, default=12345)
    e.add_argument("--log-every", type=float, default=30, help="seconds between progress lines")
    e.add_argument("--weights", default=DEFAULT_WEIGHTS)

    args = parser.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
