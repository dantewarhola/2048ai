# 2048 AI

A 2048 player that teaches itself, and a desktop app to train it, check its progress and
watch it play [2048.org](https://2048.org/) in your browser.

- **Learns by itself.** It improves by playing hundreds of thousands of games against
  itself (an n-tuple network trained with TD(0) learning).
- **Plays the corner strategy.** It keeps the biggest tile in the bottom-right corner, with
  the next biggest tiles lined up in a snake beside it, so merges chain back into the corner.
- **Looks ahead.** An expectimax search weighs every possible tile spawn before each move.
- **Plays the real game.** It reads the board from your screen, including the number on
  each tile, and presses the keys for you.

## Download the app (Windows)

The easiest way: download **`2048 AI.exe`** from the
[latest release](https://github.com/dantewarhola/2048ai/releases/latest) and double-click
it. Nothing else to install. On first launch, press **Download pretrained model** to get a
trained model, or train your own.

Windows may warn that the app is from an unknown publisher, because it isn't code-signed.
Click **More info → Run anyway**.

The app keeps its models and settings in `%LOCALAPPDATA% 2048 AI`.

## Run from source

You need:

- **Windows 10 or 11** to play in the browser. Training and evaluation also work on macOS
  and Linux.
- **Python 3.9 or newer** ([python.org](https://www.python.org/downloads/)). Tick
  *"Add python.exe to PATH"* when you install it.
- About 1 GB of free memory while training.

1. Download the project: click **Code → Download ZIP** on GitHub and unzip it, or run:
   ```
   git clone https://github.com/dantewarhola/2048ai.git
   ```
2. Double-click **`run.bat`**. The first time, it installs the two packages the app needs
   (numpy and Pillow), then opens the app.

Or, from a terminal in the project folder:
```
pip install -r requirements.txt
python app.py
```

Run from source, the app keeps models and settings in the project folder.

## Using the app

| Page | What it does |
|---|---|
| **Home** | The active model, a setup check, and shortcuts to everything else. |
| **Play** | The AI plays 2048 in your browser while the app shows the board, moves per second and thinking time. |
| **Train** | Trains the active model, until you press Stop or for a set number of games, and shows live progress. You can also run an evaluation: test games with look-ahead, no learning. |
| **Models** | Creates, picks, duplicates, renames, deletes, imports and exports models. |
| **Stats** | Live charts of training progress and the latest evaluation results. |

### Playing in the browser

1. Open [2048.org](https://2048.org/) (the **Open 2048.org** button does this) and make sure
   the whole board is visible on screen.
2. On the **Play** page, press **Start AI**, then click on the game page within 5 seconds so
   it receives the key presses.
3. When *"You win!"* appears, click **Keep going** and the AI carries on.
4. Press **Stop** in the app, or **Q** in any window, to stop.

Don't move or cover the browser window while it plays. The AI reads the board from the
screen.

### Training

New models start from scratch. Expect a few thousand games before the model plays well,
and hundreds of thousands before it plays at its best. Training saves every few minutes
and again whenever you press **Stop and save**, so you can stop and continue later.

### Sharing a trained model

A model is a 270 MB `.npy` file, which is too big for a git repository. On the
**Models** page, **Export for sharing** saves a compact `.npz` copy (about 25 MB) that
you can attach to a GitHub release. **Import** loads either format.

To have the app offer a one-click download of your model, set `PRETRAINED_URL` near the
top of `app.py` to the release file's download link. A **Download pretrained model**
button then appears on the Home and Models pages.

## Command line

Everything the app does is also available from a terminal:

```
python agent2048.py train                     # train until Ctrl+C (saves first)
python agent2048.py train --games 200000      # train for a set number of games
python agent2048.py train --corner 0          # train without the corner strategy
python agent2048.py eval --games 200          # measure the model (look-ahead depth 1)
python ai_player.py                           # play in the browser until the game is over
python ai_player.py --target 8192 --depth 3   # stop at 8192; look further ahead
python stats_viewer.py                        # the stats window on its own
python read_board.py                          # print the board it sees on screen
python game2048.py --play                     # play 2048 yourself in the terminal
```

Add `--help` to any of these for all options.

## How it works

- **`game2048.py`**: a fast game engine that runs thousands of games at once, using
  lookup tables for every possible row.
- **`agent2048.py`**: the value network and the learning loop. The network rates a board
  by looking up 4 groups of 6 cells in all 8 rotations and reflections of the board. After
  each move, the rating of the previous board is nudged toward the points earned plus the
  rating of the next board. When it plays, every move also scores a snake pattern anchored
  in the bottom-right corner, which steers it toward the classic corner strategy.
- **`ai_player.py`**: plays in the browser using expectimax search on top of the network.
  It predicts each board after its move, so only the newly spawned tile has to be read
  from the screen.
- **`read_board.py`**: finds the board on screen and reads the number printed on each
  tile, by matching its shape against every possible tile value. 2048.org draws every tile
  from 4096 up in the same dark color, so tile colors alone can't tell them apart.
- **`app.py`**: the desktop app (Tkinter). It runs the scripts above as background
  processes.
- **`theme.py`**: the app's colors, fonts and widgets.
- **`runtime.py`**: where models and settings are stored, and how the app starts the
  scripts above, both from source and from the exe.
