"""Play 2048 in the browser with the trained agent and expectimax search.

Starts read_board.py (via browser_io.BoardWatcher), and for every new
board it prints, searches for the best move and presses W/A/S/D.

Usage:
    python ai_player.py                      # depth-2 search, play until game over
    python ai_player.py --target 8192        # stop once an 8192 tile is made
    python ai_player.py --depth 1            # faster, weaker
    python ai_player.py --corner 0           # no bottom-right corner preference
By default the search favors the classic corner strategy: the biggest tile
kept in the bottom-right corner, with the next biggest lined up beside it
along a snake path (bottom row right to left, then the row above left to
right, ...), so merges chain back into the corner.

Then click on the game window so it receives the key presses.
Press Q in any window to stop.

If the "You win!" screen appears, click "Keep going" yourself; the player
waits and carries on as soon as the board is visible again.
"""

import argparse
import time

import numpy as np

from agent2048 import DEFAULT_CORNER, DEFAULT_WEIGHTS, Agent
from game2048 import slide
from browser_io import KEYS, BoardWatcher, press_key
from read_board import q_pressed, wait

ACTION_KEYS = ["W", "A", "S", "D"]  # matches game2048 actions: up, left, down, right
START_DELAY = 5.0


def to_exponents(board):
    """Tile values -> exponents (0 = empty), as the agent uses."""
    values = np.array(board, dtype=np.int64)
    exps = np.zeros(values.shape, dtype=np.int8)
    nz = values > 0
    exps[nz] = np.log2(values[nz]).round().astype(np.int8)
    return exps


BIG_EXP = 7  # 128 and up: never spawned, so the prediction knows them exactly


def reconcile(predicted, board):
    """The predicted afterstate plus the new tile seen on screen, or None.

    The screen must show exactly one new 2 or 4 in a cell the prediction left
    empty. Big tiles never spawn, so where both say "big tile" but disagree on
    which (a misread number), the prediction wins.
    """
    diff = board != predicted
    spawn = diff & (predicted == 0)
    misread = diff & (predicted >= BIG_EXP) & (board >= BIG_EXP)
    if spawn.sum() != 1 or (diff & ~spawn & ~misread).any():
        return None
    r, c = np.argwhere(spawn)[0]
    if board[r, c] not in (1, 2):
        return None
    fixed = predicted.copy()
    fixed[r, c] = board[r, c]
    return fixed


def main():
    parser = argparse.ArgumentParser(description="Play 2048 in the browser with the trained agent.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--depth", type=int, default=2, help="expectimax search depth (0 = greedy)")
    parser.add_argument("--corner", type=float, default=DEFAULT_CORNER,
                        help="weight of the bottom-right snake strategy (0 = network only)")
    parser.add_argument("--deep-empty", type=int, default=2,
                        help="search one level deeper when this many or fewer cells are empty (0 = never)")
    parser.add_argument("--target", type=int, default=0, help="stop once this tile is made (0 = play until game over)")
    parser.add_argument("--delay", type=float, default=0.0, help="extra seconds to wait after each move")
    parser.add_argument("--retry", type=float, default=1.5,
                        help="seconds without a board change before pressing the key again")
    args = parser.parse_args()

    agent = Agent.load(args.weights, args.corner)
    q_pressed()  # clear any stale "pressed since last check" state
    watcher = BoardWatcher()
    try:
        play(agent, watcher, args)
    finally:
        watcher.stop()


def play(agent, watcher, args):
    print("Waiting for read_board.py to see a 2048 board...", flush=True)
    while True:
        board, visible, _, finished = watcher.state()
        if finished:
            print("read_board.py exited.")
            return
        if board is not None and visible:
            break
        if wait(0.2):
            print("Q pressed, exiting.")
            return

    print(f"Board found. Click on the game window; first move in {START_DELAY:.0f} seconds.", flush=True)
    if wait(START_DELAY):
        print("Q pressed, exiting.")
        return

    moves = 0
    used_update = -1        # watcher update count of the board we last moved on
    predicted = None        # expected afterstate of our last move
    last_action = None
    last_press = 0.0
    retries = 0
    grace_until = None      # waiting briefly for a board that matches our prediction
    hidden_notice = False

    while True:
        if q_pressed():
            print("Q pressed, exiting.")
            return
        time.sleep(0.005)
        board, visible, updates, finished = watcher.state()
        now = time.perf_counter()
        if finished:
            print("read_board.py exited.")
            return

        if not visible:
            if not hidden_notice:
                print("Board hidden. If the 'You win!' screen is showing, click 'Keep going'; "
                      "play resumes when the board is visible.", flush=True)
                hidden_notice = True
            continue
        hidden_notice = False

        if updates == used_update:
            # No new board since our move: the key press may have been missed.
            if last_action is not None and now - last_press > args.retry:
                retries += 1
                if retries <= 3:
                    print(f"No change seen; pressing {ACTION_KEYS[last_action]} again.", flush=True)
                    press_key(KEYS[ACTION_KEYS[last_action]])
                elif retries == 4:
                    print("Still no change. Is the game window focused? Waiting...", flush=True)
                last_press = now
            continue

        exps = to_exponents(board)
        fixed = reconcile(predicted, exps) if predicted is not None else exps
        if fixed is None:
            # Possibly a frame caught mid-animation; give it a moment to settle.
            if grace_until is None:
                grace_until = now + 0.5
            if now < grace_until:
                continue
            print("Warning: board doesn't match the last move's prediction; "
                  "trusting the screen.", flush=True)
        else:
            exps = fixed
        grace_until = None

        max_tile = 2 ** int(exps.max())
        if args.target and max_tile >= args.target:
            print(f"Reached {max_tile} after {moves} moves. Done!")
            return

        empty = int((exps == 0).sum())
        depth = args.depth + (1 if args.depth and empty <= args.deep_empty else 0)
        t0 = time.perf_counter()
        action = agent.best_move(exps, depth)
        think = time.perf_counter() - t0
        if action is None:
            print(f"Game over after {moves} moves. Best tile {max_tile}.")
            return

        moves += 1
        key = ACTION_KEYS[action]
        print(f"Move {moves}: {key}  (depth {depth}, {think * 1000:.0f} ms)", flush=True)
        press_key(KEYS[key])
        predicted = slide(exps[None], action)[0][0]
        used_update = updates
        last_action = action
        last_press = time.perf_counter()
        retries = 0
        if args.delay and wait(args.delay):
            print("Q pressed, exiting.")
            return


if __name__ == "__main__":
    main()
