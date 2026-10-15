"""Lightweight, vectorized 2048 engine for running many games at once.

Implements the rules in Rules.txt, Moves.txt and Scoring.txt:
  * 4x4 board; a game starts with 2 random tiles, each a 2 (90%) or 4 (10%).
  * A move slides every tile in one direction. Equal tiles merge into one tile
    of double the value; a tile merges at most once per move, and the pair
    nearest the target edge merges first (2 2 2 _ -> 4 2 _ _).
  * A move that changes nothing is invalid: no tile spawns, no points.
  * After a valid move one tile spawns in a random empty cell (2: 90%, 4: 10%).
  * Each merge scores the value of the new tile.
  * Reaching 2048 is a win, but play may continue; the game is over when no
    move can change the board.

Boards are stored as tile exponents (0 = empty, 1 = 2, 2 = 4, ... 11 = 2048)
in an int8 array of shape (n_games, 4, 4). Moves use a precomputed lookup
table of every possible row, so a step for thousands of games is a handful of
numpy operations.

Actions: 0 = up (W), 1 = left (A), 2 = down (S), 3 = right (D).

Example:
    from game2048 import Game2048
    env = Game2048(n_games=1024, seed=0)
    boards = env.reset()
    while not env.done.all():
        actions = np.random.randint(0, 4, size=env.n)
        boards, rewards, done, info = env.step(actions)
    print(env.score.mean(), env.max_tile().max())

Command line:
    python game2048.py --play          # play one game in the terminal
    python game2048.py --bench 10000   # play N random games at once, report speed
    python game2048.py --test          # check the engine against the rule examples
"""

import argparse
import time

import numpy as np

SIZE = 4
N_ACTIONS = 4
UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3
ACTION_NAMES = ["up", "left", "down", "right"]

# Largest exponent stored. 2^17 = 131072 is the largest tile possible on a
# 4x4 board, so this never limits real play.
MAX_EXP = 17
_BASE = MAX_EXP + 1
_POW = _BASE ** np.arange(SIZE - 1, -1, -1)  # row digits -> table index

WIN_EXP = 11  # 2048


def _build_tables():
    """For every possible row, its result after a left move and points earned."""
    n_rows = _BASE ** SIZE
    rows = (np.arange(n_rows)[:, None] // _POW) % _BASE
    result = np.zeros((n_rows, SIZE), dtype=np.int8)
    score = np.zeros(n_rows, dtype=np.int64)
    for i, row in enumerate(rows.tolist()):
        tiles = [t for t in row if t]
        out = []
        gained = 0
        j = 0
        while j < len(tiles):
            # Tiles nearest the left edge are processed first, and a merged tile
            # cannot merge again, so pairs are taken greedily from the left.
            if j + 1 < len(tiles) and tiles[j] == tiles[j + 1] and tiles[j] < MAX_EXP:
                out.append(tiles[j] + 1)
                gained += 2 ** (tiles[j] + 1)
                j += 2
            else:
                out.append(tiles[j])
                j += 1
        result[i, :len(out)] = out
        score[i] = gained
    changed = (result != rows).any(axis=1)
    return result, score, changed


_LEFT_RESULT, _LEFT_SCORE, _LEFT_CHANGED = _build_tables()


def _to_left(boards, action):
    """View boards so that `action` becomes a left move."""
    if action == LEFT:
        return boards
    if action == RIGHT:
        return boards[:, :, ::-1]
    if action == UP:
        return boards.transpose(0, 2, 1)
    return boards.transpose(0, 2, 1)[:, :, ::-1]  # DOWN


def _from_left(boards, action):
    """Inverse of _to_left."""
    if action == LEFT:
        return boards
    if action == RIGHT:
        return boards[:, :, ::-1]
    if action == UP:
        return boards.transpose(0, 2, 1)
    return boards[:, :, ::-1].transpose(0, 2, 1)  # DOWN


def _row_index(boards):
    """(n, 4, 4) boards -> (n, 4) lookup-table index of each row."""
    return (boards.astype(np.int64) * _POW).sum(axis=2)


def slide(boards, action):
    """Apply one move to every board, without spawning a tile.

    Returns (afterstates, points, changed): the boards after sliding/merging,
    the points each move scores, and whether each board changed (valid move).
    """
    idx = _row_index(_to_left(boards, action))
    after = np.ascontiguousarray(_from_left(_LEFT_RESULT[idx], action))
    points = _LEFT_SCORE[idx].sum(axis=1)
    changed = _LEFT_CHANGED[idx].any(axis=1)
    return after, points, changed


class Game2048:
    """A batch of independent 2048 games advanced together.

    Attributes:
        boards: int8 (n, 4, 4) tile exponents (0 = empty).
        score:  int64 (n,) total points of each game.
        moves:  int64 (n,) number of valid moves made in each game.
        done:   bool (n,) True once a game has no valid moves left.
    """

    def __init__(self, n_games=1, seed=None):
        self.n = n_games
        self.rng = np.random.default_rng(seed)
        self.boards = np.zeros((n_games, SIZE, SIZE), dtype=np.int8)
        self.score = np.zeros(n_games, dtype=np.int64)
        self.moves = np.zeros(n_games, dtype=np.int64)
        self.done = np.zeros(n_games, dtype=bool)
        self.reset()

    def reset(self, mask=None):
        """Start new games (all, or only where `mask` is True). Returns boards."""
        if mask is None:
            mask = np.ones(self.n, dtype=bool)
        idx = np.flatnonzero(mask)
        self.boards[idx] = 0
        self.score[idx] = 0
        self.moves[idx] = 0
        self.done[idx] = False
        self._spawn(idx)
        self._spawn(idx)
        return self.boards

    def _spawn(self, idx):
        """Add one tile (2 at 90%, 4 at 10%) to a random empty cell of each game in idx."""
        if len(idx) == 0:
            return
        flat = self.boards[idx].reshape(len(idx), -1)
        empty = flat == 0
        has_empty = empty.any(axis=1)
        # Random key per cell; empty cells get keys in (0, 1], full ones -1.
        keys = np.where(empty, 1.0 - self.rng.random(flat.shape), -1.0)
        cell = keys.argmax(axis=1)
        value = np.where(self.rng.random(len(idx)) < 0.9, 1, 2).astype(np.int8)
        rows = np.flatnonzero(has_empty)
        flat[rows, cell[rows]] = value[rows]
        self.boards[idx] = flat.reshape(-1, SIZE, SIZE)

    def valid_moves(self, boards=None):
        """bool (n, 4): which actions would change each board."""
        if boards is None:
            boards = self.boards
        valid = np.zeros((len(boards), N_ACTIONS), dtype=bool)
        for a in range(N_ACTIONS):
            valid[:, a] = _LEFT_CHANGED[_row_index(_to_left(boards, a))].any(axis=1)
        return valid

    def step(self, actions):
        """Apply one action per game.

        Games that are already done are left unchanged. Returns
        (boards, rewards, done, info) where rewards are the points scored by
        this move and info["valid"] marks games whose move changed the board.
        """
        actions = np.asarray(actions)
        if actions.shape != (self.n,):
            raise ValueError(f"expected {self.n} actions, got shape {actions.shape}")

        new = self.boards.copy()
        rewards = np.zeros(self.n, dtype=np.int64)
        active = ~self.done
        for a in range(N_ACTIONS):
            sel = np.flatnonzero(active & (actions == a))
            if len(sel) == 0:
                continue
            new[sel], rewards[sel], _ = slide(self.boards[sel], a)

        valid = (new != self.boards).any(axis=(1, 2))
        self.boards = new
        self._spawn(np.flatnonzero(valid))
        self.score += rewards
        self.moves += valid
        self.done |= ~self.valid_moves().any(axis=1)
        return self.boards, rewards, self.done, {"valid": valid}

    def max_tile(self):
        """int64 (n,): value of the largest tile in each game."""
        exp = self.boards.reshape(self.n, -1).max(axis=1).astype(np.int64)
        return np.where(exp > 0, 2 ** exp, 0)

    def won(self):
        """bool (n,): games that have made a 2048 tile."""
        return self.boards.reshape(self.n, -1).max(axis=1) >= WIN_EXP

    def values(self):
        """int64 (n, 4, 4): boards as tile values (0 = empty)."""
        b = self.boards.astype(np.int64)
        return np.where(b > 0, 2 ** b, 0)

    def render(self, i=0):
        """Text picture of game i."""
        line = "+" + "+".join("-" * 7 for _ in range(SIZE)) + "+"
        out = [f"Score: {self.score[i]}", line]
        for row in self.values()[i]:
            out.append("|" + "|".join(f" {v if v else '.':>5} " for v in row) + "|")
            out.append(line)
        return "\n".join(out)


def encode_one_hot(boards):
    """float32 (n, MAX_EXP + 1, 4, 4) one-hot channels, a common network input."""
    boards = np.asarray(boards)
    return (boards[:, None, :, :] == np.arange(_BASE)[None, :, None, None]).astype(np.float32)


# ---------------------------------------------------------------- command line

def play():
    env = Game2048(1)
    keys = {"w": UP, "a": LEFT, "s": DOWN, "d": RIGHT}
    while not env.done[0]:
        print(env.render())
        key = input("Move (w/a/s/d, q to quit): ").strip().lower()
        if key == "q":
            return
        if key not in keys:
            continue
        _, _, _, info = env.step(np.array([keys[key]]))
        if not info["valid"][0]:
            print("Nothing moved.")
    print(env.render())
    print("Game over!", "You won!" if env.won()[0] else "")


def bench(n):
    env = Game2048(n, seed=0)
    start = time.perf_counter()
    steps = 0
    while not env.done.all():
        # Random play, restricted to valid moves so games finish quickly.
        valid = env.valid_moves()
        keys = np.where(valid, env.rng.random(valid.shape), -1)
        env.step(keys.argmax(axis=1))
        steps += 1
    elapsed = time.perf_counter() - start
    total_moves = env.moves.sum()
    print(f"{n} games finished in {elapsed:.2f}s ({steps} batch steps, "
          f"{total_moves / elapsed:,.0f} moves/s)")
    print(f"mean score {env.score.mean():.0f}, best score {env.score.max()}, "
          f"best tile {env.max_tile().max()}")


def selftest():
    def left(row):
        env = Game2048(1)
        env.boards[0] = 0
        env.boards[0, 0] = [int(np.log2(v)) if v else 0 for v in row]
        idx = _row_index(env.boards[:, :1])[0, 0]
        return [2 ** int(e) if e else 0 for e in _LEFT_RESULT[idx]], int(_LEFT_SCORE[idx])

    # Examples from Moves.txt, Rules.txt and Scoring.txt.
    cases = [
        ([0, 0, 2, 0], [2, 0, 0, 0], 0),
        ([2, 2, 0, 0], [4, 0, 0, 0], 4),
        ([2, 2, 4, 4], [4, 8, 0, 0], 12),
        ([2, 2, 2, 0], [4, 2, 0, 0], 4),
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([4, 2, 2, 0], [4, 4, 0, 0], 4),
        ([4, 4, 4, 0], [8, 4, 0, 0], 8),
        ([4, 4, 8, 8], [8, 16, 0, 0], 24),
    ]
    for before, after, points in cases:
        got, score = left(before)
        assert (got, score) == (after, points), (before, got, score)

    # Every direction, on the same board: column 0 holds 2,2,4,0 top to bottom.
    env = Game2048(4, seed=1)
    env.boards[:] = 0
    env.boards[:, :, 0] = [1, 1, 2, 0]
    env.boards[:, 0, 2] = 3  # an 8 in the top row: 2 _ 8 _
    before = env.boards.copy()
    _, rewards, _, info = env.step(np.array([UP, LEFT, DOWN, RIGHT]))
    assert info["valid"].all()
    assert list(rewards) == [4, 0, 4, 0], rewards
    spawned = (env.boards != 0).sum(axis=(1, 2)) - [3, 4, 3, 4]
    assert (spawned == 1).all(), spawned
    assert list(env.boards[0, :2, 0]) == [2, 2]          # up: 2+2=4 on top, then 4
    assert list(env.boards[2, 2:, 0]) == [2, 2]          # down: bottom pair merges, 4 4 at bottom
    assert list(env.boards[3, 0, 2:]) == [1, 3]          # right: 2 8 slide right

    # Invalid move: nothing changes and no tile spawns.
    env = Game2048(1, seed=2)
    env.boards[0] = 0
    env.boards[0, 0, 0] = 1
    _, rewards, _, info = env.step(np.array([LEFT]))
    assert not info["valid"][0] and rewards[0] == 0
    assert (env.boards[0] != 0).sum() == 1

    # Game over: full board, no equal neighbours. Full board with a pair: not over.
    env = Game2048(2, seed=3)
    checker = np.array([[1, 2, 1, 2], [2, 1, 2, 1]] * 2, dtype=np.int8)
    env.boards[:] = checker
    env.boards[1, 3, 3] = 2  # now equal to its left neighbour
    assert list(env.valid_moves().any(axis=1)) == [False, True]

    # Starting boards: exactly 2 tiles, each a 2 or a 4, roughly 90/10.
    env = Game2048(20000, seed=4)
    tiles = env.boards[env.boards > 0]
    assert (tiles.size == 40000) and set(np.unique(tiles)) <= {1, 2}
    assert abs((tiles == 1).mean() - 0.9) < 0.01
    print("All rule checks passed.")


def main():
    parser = argparse.ArgumentParser(description="Vectorized 2048 engine.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--play", action="store_true", help="play one game in the terminal")
    group.add_argument("--bench", type=int, metavar="N", help="play N random games at once")
    group.add_argument("--test", action="store_true", help="check the engine against the rules")
    args = parser.parse_args()
    if args.bench:
        bench(args.bench)
    elif args.test:
        selftest()
    else:
        play()


if __name__ == "__main__":
    main()
