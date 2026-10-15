"""2048 AI - desktop app.

One window for everything in this project:

    Home     overview of the active model and quick actions
    Play     let the AI play 2048 in your browser, with a live view of the board
    Train    train the active model (and measure it with an evaluation run)
    Models   pick, create, import, export, rename and delete weight files
    Stats    the live training dashboard (stats_viewer.py)

Run it with:
    python app.py
or double-click run.bat on Windows.

Training, evaluation and play each run as a separate Python process (the same
scripts you can run from a terminal); the app shows their output and stops
them through a "stop file", so training always saves before it exits.
"""

import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import webbrowser
import codecs
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

from agent2048 import (DEFAULT_CORNER, DEFAULT_WEIGHTS, STOP_FILE_ENV, eval_path, format_duration,
                       load_compact, read_json, save_compact, stats_path, write_json)
import theme
from runtime import DATA_DIR, FROZEN, run_script, script_command
from stats_viewer import StatsDashboard, fmt_ago
from theme import (BAD, BG, BORDER, BORDER_STRONG, BRAND, BRAND_DEEP, BRAND_SOFT, CONSOLE_BAR,
                   CONSOLE_BG, CONSOLE_MUTED, CONSOLE_TEXT, FONT, FONT_BIG, FONT_BOLD, FONT_CAPS, FONT_H1,
                   FONT_H2, FONT_HERO, FONT_MONO, FONT_SMALL, FONT_SMALL_BOLD, FONT_STAT, GOOD,
                   MUTED, PANEL, SHADOW, SIDEBAR, SIDEBAR_ACTIVE, SIDEBAR_HOVER, SIDEBAR_MUTED,
                   SIDEBAR_TEXT, SUBTLE, TEXT, TEXT_2, WARN, Card, FlatButton, Segmented, SliderField,
                   StatusPill, WrapLabel, divider, field, px, round_rect)

APP_NAME = "2048 AI"
# From the repo this is the project folder; in the packaged exe, %LOCALAPPDATA%\2048 AI.
HERE = DATA_DIR
MODELS_DIR = os.path.join(HERE, "models")
SETTINGS_FILE = os.path.join(HERE, "app_settings.json")
GAME_URL = "https://2048.org/"
# A compact pretrained model (.npz, made with Models > Export) to offer new
# users, e.g. a GitHub release asset. Leave empty to hide the download button.
PRETRAINED_URL = "https://github.com/dantewarhola/2048ai/releases/download/v1.0.0/agent_weights.npz"
IS_WINDOWS = sys.platform == "win32"

# ---------------------------------------------------------------- look
# Colors, fonts and the shared widgets live in theme.py.

GOLD = BRAND

# 2048.org tile colors: value -> (background, text).
TILE_COLORS = {
    0: ("#cdc1b4", "#cdc1b4"), 2: ("#eee4da", "#776e65"), 4: ("#ede0c8", "#776e65"),
    8: ("#f2b179", "#f9f6f2"), 16: ("#f59563", "#f9f6f2"), 32: ("#f67c5f", "#f9f6f2"),
    64: ("#f65e3b", "#f9f6f2"), 128: ("#edcf72", "#f9f6f2"), 256: ("#edcc61", "#f9f6f2"),
    512: ("#edc850", "#f9f6f2"), 1024: ("#edc53f", "#f9f6f2"), 2048: ("#edc22e", "#f9f6f2"),
}
SUPER_TILE = ("#3c3a32", "#f9f6f2")  # 4096 and up


def tile_colors(value):
    return TILE_COLORS.get(value, SUPER_TILE)


# ---------------------------------------------------------------- settings

DEFAULT_SETTINGS = {
    "model": DEFAULT_WEIGHTS,
    "new_models": [],  # created in the app but not trained yet (no file on disk)
    "play": {"depth": 2, "corner": DEFAULT_CORNER, "target": 0, "delay": 0.0},
    "train": {"games": 100000, "until_stop": True, "corner": DEFAULT_CORNER, "alpha": 0.1,
              "batch": 256, "save_minutes": 5},
    "eval": {"games": 200, "depth": 1},
}


def load_settings():
    data = read_json(SETTINGS_FILE) or {}
    settings = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_SETTINGS.items()}
    for key, value in data.items():
        if isinstance(settings.get(key), dict) and isinstance(value, dict):
            settings[key].update(value)
        else:
            settings[key] = value
    settings["model"] = abs_path(settings["model"])
    settings["new_models"] = [abs_path(p) for p in settings["new_models"]]
    return settings


def save_settings(settings):
    data = dict(settings)
    data["model"] = rel_path(settings["model"])
    data["new_models"] = [rel_path(p) for p in settings["new_models"]]
    try:
        write_json(SETTINGS_FILE, data)
    except OSError:
        pass


def rel_path(path):
    """Paths inside the project are stored relative, so the folder can be moved."""
    try:
        rel = os.path.relpath(path, HERE)
    except ValueError:  # another drive
        return path
    return path if rel.startswith("..") else rel


def abs_path(path):
    return os.path.normpath(path if os.path.isabs(path) else os.path.join(HERE, path))


# ---------------------------------------------------------------- models

def model_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def sidecar_files(path):
    """Files that belong with a weights file: info, live stats, latest eval."""
    base = os.path.splitext(path)[0]
    return [base + ".json", stats_path(path), eval_path(path)]


def model_info(path):
    return read_json(os.path.splitext(path)[0] + ".json") or {}


def games_trained(path):
    """Training games so far, including any since the last checkpoint."""
    saved = model_info(path).get("games_trained", 0)
    live = (read_json(stats_path(path)) or {}).get("games_trained", 0)
    return max(saved, live)


def find_models(extra=()):
    """Weights files in the project folder and models/, plus `extra` paths."""
    found = []
    for folder in (HERE, MODELS_DIR):
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                if name.endswith(".npy") and ".tmp" not in name:
                    found.append(os.path.normpath(os.path.join(folder, name)))
    for path in extra:
        if os.path.normpath(path) not in found:
            found.append(os.path.normpath(path))
    return found


def fmt_size(n):
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


def valid_model_name(name):
    return bool(name) and re.fullmatch(r"[A-Za-z0-9 _\-.]+", name) and not name.startswith(".")


# ---------------------------------------------------------------- background jobs

class Job:
    """One of the project's scripts running as a child process.

    Output is split into lines ("line" events) and carriage-return progress
    updates ("status" events), delivered on the Tk thread through App.events.
    """

    def __init__(self, app, name, script_args, on_line=None, on_status=None, on_exit=None):
        self.app = app
        self.name = name
        self.on_line, self.on_status, self.on_exit = on_line, on_status, on_exit
        self.stop_file = os.path.join(tempfile.gettempdir(), f"ai2048_stop_{os.getpid()}_{name}")
        if os.path.exists(self.stop_file):
            os.remove(self.stop_file)
        env = os.environ.copy()
        env[STOP_FILE_ENV] = self.stop_file
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        flags = subprocess.CREATE_NO_WINDOW if IS_WINDOWS else 0
        self.proc = subprocess.Popen(script_command(*script_args), cwd=HERE, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, creationflags=flags)
        self.stopping = False
        self.started = time.time()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pending = ""
        while True:
            chunk = self.proc.stdout.read1(4096)
            if not chunk:
                break
            pending += decoder.decode(chunk)
            while True:
                m = re.search(r"\r\n|\r(?=.)|\n", pending, re.S)  # a lone \r at the end may start \r\n
                if not m:
                    break
                text, sep, pending = pending[:m.start()], m.group(), pending[m.end():]
                if sep != "\r":
                    self.app.events.put((self, "line", text.rstrip()))
                elif text.strip():  # "\r" redraws a progress line in place
                    self.app.events.put((self, "status", text.strip()))
        if pending.strip():
            self.app.events.put((self, "line", pending))
        self.app.events.put((self, "exit", self.proc.wait()))

    @property
    def running(self):
        return self.proc.poll() is None

    def stop(self, grace=60.0):
        """Ask the script to stop (training saves first); force it after `grace` seconds."""
        if not self.running:
            return
        self.stopping = True
        try:
            open(self.stop_file, "w").close()
        except OSError:
            self.kill()
            return
        self.app.after(int(grace * 1000), lambda: self.running and self.kill())

    def kill(self):
        """End the process and anything it started (ai_player runs read_board.py)."""
        if not self.running:
            return
        if IS_WINDOWS:
            subprocess.run(["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            self.proc.kill()

    def cleanup(self):
        try:
            os.remove(self.stop_file)
        except OSError:
            pass


# ---------------------------------------------------------------- widgets
# Generic widgets (Card, FlatButton, SliderField, ...) are in theme.py.

class LogView(Card):
    """Console-style output with a one-line status bar for progress updates."""

    MAX_LINES = 1500

    def __init__(self, parent, height=12, title="Output"):
        super().__init__(parent, padx=0, pady=0, bg=CONSOLE_BG, border=CONSOLE_BG, lift=False)
        head = tk.Frame(self, bg=CONSOLE_BG)
        head.pack(fill="x", padx=px(14), pady=(px(10), px(4)))
        tk.Label(head, text=title.upper(), font=FONT_CAPS, bg=CONSOLE_BG, fg=CONSOLE_MUTED).pack(side="left")
        clear = tk.Label(head, text="Clear", font=FONT_SMALL, bg=CONSOLE_BG, fg=CONSOLE_MUTED, cursor="hand2")
        clear.pack(side="right")
        clear.bind("<Button-1>", lambda e: self.clear())
        clear.bind("<Enter>", lambda e: clear.config(fg=CONSOLE_TEXT))
        clear.bind("<Leave>", lambda e: clear.config(fg=CONSOLE_MUTED))
        # No scrollbar: the mouse wheel scrolls, and it keeps the console clean.
        self.text = tk.Text(self, height=height, bg=CONSOLE_BG, fg=CONSOLE_TEXT, font=FONT_MONO,
                            relief="flat", wrap="word", insertbackground=CONSOLE_TEXT,
                            highlightthickness=0, padx=px(14), pady=0, spacing1=1, spacing3=1,
                            selectbackground="#3a352f")
        self.text.config(state="disabled")
        self.text.pack(fill="both", expand=True)
        self.status_label = tk.Label(self, text="", font=FONT_MONO, bg=CONSOLE_BAR, fg=BRAND,
                                     anchor="w", padx=px(14), pady=px(5))
        self.status_label.pack(fill="x", pady=(px(6), 0))
        self.status_label.pack_forget()

    def write(self, line):
        at_end = self.text.yview()[1] > 0.999
        self.text.config(state="normal")
        self.text.insert("end", line + "\n")
        lines = int(self.text.index("end-1c").split(".")[0])
        if lines > self.MAX_LINES:
            self.text.delete("1.0", f"{lines - self.MAX_LINES}.0")
        self.text.config(state="disabled")
        if at_end:
            self.text.see("end")

    def status(self, text):
        self.status_label.config(text=text)
        if text and not self.status_label.winfo_ismapped():
            self.status_label.pack(fill="x", pady=(px(6), 0))

    def clear(self):
        self.text.config(state="normal")
        self.text.delete("1.0", "end")
        self.text.config(state="disabled")
        self.status_label.config(text="")
        self.status_label.pack_forget()


class BoardView(tk.Canvas):
    """A 2048 board drawn in the game's colors."""

    def __init__(self, parent, size=360):
        super().__init__(parent, width=size, height=size, bg=theme.bg_of(parent), highlightthickness=0)
        self.board = None
        self.placeholder = ("Waiting for the game", "")
        self.bind("<Configure>", lambda e: self.draw())

    def show(self, board):
        self.board = board
        self.draw()

    def set_placeholder(self, title, detail=""):
        self.placeholder = (title, detail)
        if not self.board:
            self.draw()

    def draw(self):
        self.delete("all")
        size = min(self.winfo_width(), self.winfo_height()) - px(3)
        if size < 40:
            return
        ox = (self.winfo_width() - size) / 2
        gap = size * 0.03
        cell = (size - 5 * gap) / 4
        round_rect(self, ox, px(3), ox + size, size + px(3), size * 0.035, fill=SHADOW, outline="")
        round_rect(self, ox, 0, ox + size, size, size * 0.035, fill="#bbada0", outline="")
        for r in range(4):
            for c in range(4):
                v = self.board[r][c] if self.board else 0
                v = v or 0
                bg, fg = tile_colors(v)
                x = ox + gap + c * (cell + gap)
                y = gap + r * (cell + gap)
                round_rect(self, x, y, x + cell, y + cell, cell * 0.08, fill=bg, outline="")
                if v:
                    digits = len(str(v))
                    scale = {1: 0.45, 2: 0.42, 3: 0.36, 4: 0.28}.get(digits, 0.22)
                    self.create_text(x + cell / 2, y + cell / 2, text=str(v), fill=fg,
                                     font=("Segoe UI Black", max(8, int(cell * scale * 0.75))))
        if not self.board:
            # A frosted panel over the empty grid, like the game's own overlays.
            title, detail = self.placeholder
            w, h = size * 0.74, size * (0.26 if detail else 0.17)
            x0, y0 = ox + (size - w) / 2, (size - h) / 2
            round_rect(self, x0, y0, x0 + w, y0 + h, px(10), fill="#f9f6f2", outline="")
            cy = size / 2 - (px(9) if detail else 0)
            self.create_text(ox + size / 2, cy, text=title, fill="#776e65", font=FONT_H2)
            if detail:
                self.create_text(ox + size / 2, cy + px(22), text=detail, fill="#9a8f84", font=FONT_SMALL,
                                 width=w - px(20), justify="center")


class Stepper(tk.Frame):
    """Numbered steps joined by a line; each is to do, current, done or a warning."""

    def __init__(self, parent, steps):
        """steps: [(title, description)]. self.slots[i] is a frame under step i's text."""
        bg = theme.bg_of(parent)
        super().__init__(parent, bg=bg)
        self.columnconfigure(1, weight=1)
        self.dots, self.lines, self.titles, self.slots = [], [], [], []
        d = px(26)
        for i, (title, desc) in enumerate(steps):
            dot = tk.Canvas(self, width=d, height=d, bg=bg, highlightthickness=0)
            dot.grid(row=2 * i, column=0, sticky="n")
            last = i == len(steps) - 1
            line = tk.Frame(self, bg=BORDER, width=px(2))
            line.grid(row=2 * i + 1, column=0, sticky="ns", pady=px(3))
            if last:
                line.grid_remove()
            head = tk.Label(self, text=title, font=FONT_BOLD, bg=bg, fg=TEXT, anchor="w")
            head.grid(row=2 * i, column=1, sticky="ew", padx=(px(12), 0), pady=(px(3), 0))
            body = tk.Frame(self, bg=bg)
            body.grid(row=2 * i + 1, column=1, sticky="ew", padx=(px(12), 0),
                      pady=(px(1), 0 if last else px(12)))
            WrapLabel(body, text=desc, font=FONT_SMALL, bg=bg, fg=MUTED, anchor="w",
                      justify="left").pack(fill="x")
            slot = tk.Frame(body, bg=bg)
            slot.pack(fill="x")
            self.dots.append(dot)
            self.lines.append(line)
            self.titles.append(head)
            self.slots.append(slot)
        self.set_states(["todo"] * len(steps))

    def set_states(self, states):
        for i, state in enumerate(states):
            c, d = self.dots[i], px(26)
            c.delete("all")
            fill, outline, mark, mark_fg = {
                "done": (GOOD, GOOD, "✓", "#ffffff"),
                "current": (BRAND, BRAND, str(i + 1), "#1f1c18"),
                "warn": (WARN, WARN, "!", "#ffffff"),
                "todo": (PANEL, BORDER_STRONG, str(i + 1), MUTED),
            }[state]
            c.create_oval(1, 1, d - 1, d - 1, fill=fill, outline=outline, width=1.5)
            c.create_text(d / 2, d / 2, text=mark, fill=mark_fg, font=FONT_SMALL_BOLD)
            self.titles[i].config(fg=MUTED if state == "todo" else TEXT)
            self.lines[i].config(bg=GOOD if state == "done" else BORDER)


def logo(parent, bg, size=44):
    c = tk.Canvas(parent, width=size, height=size, bg=bg, highlightthickness=0)
    round_rect(c, 1, 1, size - 1, size - 1, size * 0.14, fill=GOLD, outline="")
    c.create_text(size / 2, size / 2, text="2048", fill="#f9f6f2",
                  font=("Segoe UI Black", int(size * 0.22)))
    return c


def make_icon():
    """A small gold tile for the title bar."""
    img = tk.PhotoImage(width=32, height=32)
    img.put(GOLD, to=(0, 0, 32, 32))
    img.put("#f9f6f2", to=(8, 13, 24, 19))
    return img


# ---------------------------------------------------------------- pages

class Page(tk.Frame):
    title = ""
    subtitle = ""

    def __init__(self, app):
        super().__init__(app.content, bg=BG)
        self.app = app
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x", padx=px(32), pady=(px(26), px(18)))
        # Right side of the header, for a page's main status or actions.
        self.header_actions = tk.Frame(head, bg=BG)
        self.header_actions.pack(side="right", anchor="s")
        titles = tk.Frame(head, bg=BG)
        titles.pack(side="left", fill="x", expand=True)
        tk.Label(titles, text=self.title, font=FONT_H1, bg=BG, fg=TEXT, anchor="w").pack(fill="x")
        if self.subtitle:
            tk.Label(titles, text=self.subtitle, font=FONT, bg=BG, fg=TEXT_2, anchor="w").pack(fill="x", pady=(2, 0))
        self.body = tk.Frame(self, bg=BG)
        self.body.pack(fill="both", expand=True, padx=px(32), pady=(0, px(24)))

    def on_show(self):
        pass

    def on_model_changed(self):
        pass

    def on_jobs_changed(self):
        pass


class HomePage(Page):
    title = "Welcome"
    subtitle = "A self-taught 2048 player you can train, measure and watch play in your browser."

    def __init__(self, app):
        super().__init__(app)
        hero = Card(self.body, padx=24, pady=20)
        hero.pack(fill="x")
        row = tk.Frame(hero, bg=PANEL)
        row.pack(fill="x")
        logo(row, PANEL, px(76)).pack(side="left", padx=(0, 18))
        txt = tk.Frame(row, bg=PANEL)
        txt.pack(side="left", fill="x", expand=True)
        tk.Label(txt, text=APP_NAME, font=FONT_HERO, bg=PANEL, fg=TEXT, anchor="w").pack(fill="x")
        WrapLabel(txt, anchor="w", justify="left", font=FONT, bg=PANEL, fg=MUTED,
                 text="An n-tuple network learns the value of 2048 positions by playing millions of "
                      "games against itself. When it plays, an expectimax search looks ahead and a "
                      "corner strategy keeps the biggest tile in the bottom-right corner, with the "
                      "next ones chained beside it.").pack(fill="x")

        mid = tk.Frame(self.body, bg=BG)
        mid.pack(fill="x", pady=16)
        mid.columnconfigure(0, weight=1)
        mid.columnconfigure(1, weight=1)

        # ---- active model
        self.model_card = Card(mid, "Active model")
        self.model_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.model_name = tk.Label(self.model_card, font=FONT_BIG, bg=PANEL, fg=TEXT, anchor="w")
        self.model_name.pack(fill="x", pady=(4, 0))
        self.model_lines = tk.Label(self.model_card, font=FONT, bg=PANEL, fg=MUTED, anchor="w",
                                    justify="left")
        self.model_lines.pack(fill="x", pady=(2, 10))
        btns = tk.Frame(self.model_card, bg=PANEL)
        btns.pack(fill="x")
        FlatButton(btns, "Change model", lambda: app.show("models"), kind="secondary").pack(side="left")
        self.download_btn = FlatButton(btns, "Download pretrained model", self.download, kind="gold")

        # ---- setup checks
        checks = Card(mid, "Ready to go?")
        checks.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        self.check_frame = tk.Frame(checks, bg=PANEL)
        self.check_frame.pack(fill="x", pady=(4, 0))

        # ---- quick actions
        actions = tk.Frame(self.body, bg=BG)
        actions.pack(fill="x")
        for i in range(3):
            actions.columnconfigure(i, weight=1)
        items = [
            ("Play", "Let the AI play 2048.org in your browser while you watch.", "play", "gold"),
            ("Train", "Teach the model with self-play. Stop any time; progress is saved.", "train", "primary"),
            ("Stats", "Live charts of training progress and evaluation results.", "stats", "primary"),
        ]
        for i, (name, desc, page, kind) in enumerate(items):
            card = Card(actions, name, desc)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            FlatButton(card, f"Open {name}", lambda p=page: app.show(p), kind=kind).pack(anchor="w", pady=(6, 0))

    def on_show(self):
        self.refresh()

    def on_model_changed(self):
        self.refresh()

    def refresh(self):
        path = self.app.model
        self.model_name.config(text=model_name(path))
        exists = os.path.exists(path)
        info = model_info(path) if exists else {}
        ev = read_json(eval_path(path)) if exists else None
        lines = []
        if not exists:
            lines.append("Not trained yet - open Train to start it from scratch.")
        else:
            lines.append(f"{games_trained(path):,} training games")
            if ev:
                lines.append(f"Evaluation: avg score {ev.get('avg_score', 0):,.0f}, best tile "
                             f"{ev.get('best_tile', 0):,} ({ev.get('games', 0)} games, {fmt_ago(ev.get('time'))})")
            else:
                lines.append("Not evaluated yet")
        self.model_lines.config(text="\n".join(lines))
        if PRETRAINED_URL:
            self.download_btn.pack(side="left", padx=(8, 0))

        for w in self.check_frame.winfo_children():
            w.destroy()
        any_model = any(os.path.exists(p) for p in find_models())
        checks = [
            (True, "Everything it needs is built in" if FROZEN else
             f"Python {sys.version.split()[0]} with numpy and Pillow"),
            (IS_WINDOWS, "Windows (needed to press keys in the browser)" if IS_WINDOWS else
             "Playing in the browser needs Windows; training works everywhere"),
            (any_model, "A trained model is available" if any_model else
             "No trained model yet: train one" + (" or download the pretrained one" if PRETRAINED_URL else "")),
        ]
        for ok, text in checks:
            row = tk.Frame(self.check_frame, bg=PANEL)
            row.pack(fill="x", pady=2)
            tk.Label(row, text="✓" if ok else "!", width=2, font=FONT_BOLD, bg=PANEL,
                     fg=GOOD if ok else BAD).pack(side="left")
            tk.Label(row, text=text, font=FONT, bg=PANEL, fg=TEXT, anchor="w").pack(side="left")

    def download(self):
        self.app.pages["models"].download_pretrained()


class PlayPage(Page):
    title = "Play"
    subtitle = "Watch the AI play 2048.org in your browser. It reads the board from your screen and presses the keys."

    MOVE_RE = re.compile(r"Move (\d+): (\w)\s+\(depth (\d+), (\d+) ms\)")
    DEPTHS = [(0, "Instant"), (1, "Fast"), (2, "Balanced"), (3, "Deep")]
    DEPTH_HINTS = {0: "No look-ahead. The fastest, and the weakest.",
                   1: "Looks one tile spawn and move ahead.",
                   2: "Strong and quick. Recommended.",
                   3: "The strongest, but each move takes longer."}
    STATS = ("Moves", "Highest tile", "Moves / second", "Thinking time", "Running for")

    # phase -> (pill text, pill tone, pulse, step states)
    PHASES = {
        "idle": ("Ready", "neutral", False, ("current", "todo", "todo", "todo")),
        "loading": ("Starting", "info", True, ("current", "done", "todo", "todo")),
        "searching": ("Looking for the board", "info", True, ("current", "done", "todo", "todo")),
        "found": ("Get ready", "brand", True, ("done", "done", "current", "todo")),
        "playing": ("Playing", "good", True, ("done", "done", "done", "current")),
        "hidden": ("Paused", "warn", False, ("done", "done", "done", "warn")),
        "stalled": ("Not responding", "warn", False, ("done", "done", "warn", "todo")),
        "stopping": ("Stopping", "neutral", True, ("done", "done", "done", "current")),
        "over": ("Game over", "neutral", False, ("done", "done", "done", "done")),
        "reached": ("Target reached", "good", False, ("done", "done", "done", "done")),
        "stopped": ("Stopped", "neutral", False, ("current", "todo", "todo", "todo")),
        "error": ("Error", "bad", False, ("current", "todo", "todo", "todo")),
    }
    MESSAGES = {
        "idle": "Open the game, then press Start AI.",
        "loading": "Loading the model...",
        "searching": "Looking for a 2048 board on screen. Open the game and keep the whole board visible.",
        "found": "Board found. Click on the game page now.",
        "playing": "The AI is playing. Press Stop, or Q in any window, to end.",
        "hidden": "Board hidden. If \"You win!\" is showing, click Keep going and play resumes.",
        "stalled": "The game isn't reacting. Click on the game page so it gets the key presses.",
        "stopping": "Stopping...",
        "stopped": "Stopped. Press Start AI to play again.",
        "error": "The AI player stopped with an error. See the activity log below.",
    }
    PLACEHOLDERS = {  # what the empty board says
        "idle": ("Ready when you are", "Press Start AI to begin"),
        "stopped": ("Ready when you are", "Press Start AI to begin"),
        "error": ("Something went wrong", "See the activity log"),
        "loading": ("Loading the model", "This takes a few seconds"),
        "searching": ("Looking for the board", "Keep the whole game board visible"),
        "found": ("Board found", "Click on the game page now"),
    }
    IDLE = ("idle", "stopped", "error", "over", "reached")

    def __init__(self, app):
        super().__init__(app)
        s = app.settings["play"]
        self.depth = tk.IntVar(value=int(s["depth"]))
        self.corner = tk.DoubleVar(value=s["corner"])
        self.target = tk.StringVar(value=self.target_label(s["target"]))
        self.delay = tk.DoubleVar(value=s["delay"])
        self.job = None
        self.rows = []
        self.moves = 0
        self.think_ms = []
        self.best = 0
        self.drawn_best = None
        self.phase = "idle"
        self.countdown_end = None

        self.pill = StatusPill(self.header_actions)
        self.pill.pack(side="right")

        self.body.columnconfigure(0, weight=0, minsize=px(360))
        self.body.columnconfigure(1, weight=1)
        self.body.rowconfigure(0, weight=1)

        # ---- left: the steps, then the settings (scrolls in a short window)
        left = theme.ScrollColumn(self.body, width=px(372))
        left.grid(row=0, column=0, sticky="nsew", padx=(0, px(20)))
        col = left.inner

        guide = Card(col, "Get started")
        guide.pack(fill="x")
        self.steps = Stepper(guide, [
            ("Open the game", "Open 2048.org and keep the whole board visible on screen."),
            ("Start the AI", "Press Start AI. It loads the model and finds the board."),
            ("Click the game page", "Do it within 5 seconds, so the game gets the key presses."),
            ("Watch it play", "If \"You win!\" appears, click Keep going. Stop any time, or press Q."),
        ])
        self.steps.pack(fill="x", pady=(px(12), 0))
        FlatButton(self.steps.slots[0], "Open 2048.org  ↗", lambda: webbrowser.open(GAME_URL),
                   kind="secondary", small=True).pack(anchor="w", pady=(px(8), 0))

        opts = Card(col, "AI settings")
        opts.pack(fill="x", pady=(px(16), 0))
        depth_row = field(opts, "Search depth")
        Segmented(depth_row, self.depth, self.DEPTHS, command=self.on_depth).pack(fill="x")
        self.depth_hint = tk.Label(depth_row, font=FONT_SMALL, bg=PANEL, fg=MUTED, anchor="w")
        self.depth_hint.pack(fill="x", pady=(px(5), 0))
        self.on_depth()
        SliderField(field(opts, "Corner strategy",
                          "How strongly it keeps the biggest tile in the bottom-right corner. 0 is off."),
                    self.corner, 0, 300)
        ttk.Combobox(field(opts, "Stop when this tile is made"), textvariable=self.target, state="readonly",
                     values=[self.target_label(t) for t in (0, 2048, 4096, 8192, 16384, 32768)]
                     ).pack(fill="x")
        SliderField(field(opts, "Pause after each move", "Raise this if the game misses key presses."),
                    self.delay, 0, 1, "{:.2f} s")
        self.settings_note = tk.Label(opts, text="Changes apply the next time you start.", font=FONT_SMALL,
                                      bg=PANEL, fg=WARN, anchor="w")

        # ---- right: the live game, then the activity log
        right = tk.Frame(self.body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        live = Card(right, padx=22, pady=20)
        live.grid(row=0, column=0, sticky="nsew")
        head = tk.Frame(live, bg=PANEL)
        head.pack(fill="x")
        self.run_btn = FlatButton(head, "▶  Start AI", self.toggle, kind="gold")
        self.run_btn.pack(side="right", anchor="n", padx=(px(16), 0))
        titles = tk.Frame(head, bg=PANEL)
        titles.pack(side="left", fill="x", expand=True)
        tk.Label(titles, text="Live game", font=FONT_H2, bg=PANEL, fg=TEXT, anchor="w").pack(fill="x")
        self.message = WrapLabel(titles, font=FONT, bg=PANEL, fg=TEXT_2, anchor="w", justify="left")
        self.message.pack(fill="x", pady=(px(2), 0))
        divider(live, pady=(px(16), px(18)))

        inner = tk.Frame(live, bg=PANEL)
        inner.pack(fill="both", expand=True)
        self.board_view = BoardView(inner, px(340))
        self.board_view.pack(side="left", anchor="n")
        nums = tk.Frame(inner, bg=PANEL)
        nums.pack(side="left", fill="both", expand=True, padx=(px(26), 0))
        self.num = {}
        for k, name in enumerate(self.STATS):
            if k:
                tk.Frame(nums, bg=BORDER, height=1).pack(fill="x", pady=px(7))
            tk.Label(nums, text=name.upper(), font=FONT_CAPS, bg=PANEL, fg=MUTED, anchor="w").pack(fill="x")
            if name == "Highest tile":  # drawn as a little tile in the game's colors
                lab = tk.Canvas(nums, height=px(34), width=px(96), bg=PANEL, highlightthickness=0)
                lab.pack(anchor="w", pady=(px(3), 0))
            else:
                # A fixed width (in digits) so values like "1:02:33" always fit.
                lab = tk.Label(nums, text="-", font=FONT_STAT, bg=PANEL, fg=TEXT, anchor="w", width=7)
                lab.pack(fill="x")
            self.num[name] = lab
        self.draw_best()
        # In a narrow window the board shrinks so the numbers keep their room.
        inner.bind("<Configure>", lambda e: self.fit_board(e.width, nums))

        self.log = LogView(right, height=6, title="Activity")
        self.log.grid(row=1, column=0, sticky="nsew", pady=(px(16), 0))
        self.set_phase("idle")
        self.tick()

    def fit_board(self, width, nums):
        size = max(px(160), min(px(340), width - nums.winfo_reqwidth() - px(26)))
        if size != self.board_view.winfo_reqwidth():
            self.board_view.config(width=size, height=size)

    def on_depth(self):
        self.depth_hint.config(text=self.DEPTH_HINTS.get(self.depth.get(), ""))

    @staticmethod
    def target_label(t):
        return "Never - play until the game is over" if not t else f"{t:,}"

    def target_value(self):
        text = self.target.get().replace(",", "")
        return int(text) if text.isdigit() else 0

    def save(self):
        self.app.settings["play"] = {"depth": int(self.depth.get()), "corner": round(self.corner.get()),
                                     "target": self.target_value(), "delay": round(self.delay.get(), 2)}

    # ---- running

    def toggle(self):
        if self.job and self.job.running:
            self.stop()
        else:
            self.start()

    def start(self):
        if not os.path.exists(self.app.model):
            messagebox.showinfo(APP_NAME, f"The model '{model_name(self.app.model)}' hasn't been trained "
                                          "yet. Train it first, or pick another one under Models.")
            return
        self.save()
        p = self.app.settings["play"]
        self.rows, self.moves, self.think_ms, self.best = [], 0, [], 0
        self.board_view.show(None)
        for name, lab in self.num.items():
            if name != "Highest tile":
                lab.config(text="-")
        self.draw_best()
        self.log.clear()
        self.job = self.app.start_job(
            "play", ["ai_player.py", "--weights", self.app.model, "--depth", str(p["depth"]),
                     "--corner", str(p["corner"]), "--target", str(p["target"]), "--delay", str(p["delay"])],
            on_line=self.on_line, on_exit=self.on_exit)
        self.set_phase("loading" if self.job else "idle")

    def stop(self):
        if self.job:
            self.job.stop(grace=5)
            self.set_phase("stopping")

    def set_phase(self, phase, message=None):
        """Move the page to a stage of the flow: pill, steps, message, board and button follow."""
        self.phase = phase
        text, tone, pulse, states = self.PHASES[phase]
        self.pill.set(text, tone, pulse)
        self.steps.set_states(states)
        if message is None:
            message = self.MESSAGES.get(phase, "")
        if not IS_WINDOWS and phase == "idle":
            message = "Playing in the browser needs Windows."
        self.message.config(text=message)
        if phase in self.PLACEHOLDERS:
            self.board_view.set_placeholder(*self.PLACEHOLDERS[phase])
        self.countdown_end = time.time() + 5 if phase == "found" else None
        self.settings_note.pack_forget()
        if phase not in self.IDLE:
            self.settings_note.pack(fill="x", pady=(px(12), 0))
        self.update_buttons()

    def on_line(self, line):
        text = line.strip()
        if text.startswith("|"):
            self.rows.append([self.parse_cell(c.strip()) for c in text.strip("|").split("|")])
            if len(self.rows) == 4:
                self.board_view.show(self.rows)
                self.best = max([self.best] + [v or 0 for row in self.rows for v in row])
                self.rows = []
            return
        if not text or text.startswith("+"):
            return
        m = self.MOVE_RE.match(text)
        if m:
            self.moves = int(m.group(1))
            self.think_ms.append(int(m.group(4)))
            self.think_ms = self.think_ms[-50:]
            if self.phase not in ("playing", "stopping"):
                self.set_phase("playing")
            return
        self.log.write(text)
        if self.phase == "stopping":
            return
        if text.startswith("Waiting for read_board"):
            self.set_phase("searching")
        elif text.startswith("Board found"):
            self.set_phase("found")
        elif text.startswith("Board hidden"):
            self.set_phase("hidden")
        elif text.startswith(("No change seen", "Still no change")):
            self.set_phase("stalled")
        elif text.startswith("Game over"):
            self.set_phase("over", text)
        elif text.startswith("Reached"):
            self.set_phase("reached", text)
        elif text.startswith("Warning"):
            self.log.status(text)

    @staticmethod
    def parse_cell(text):
        if text == ".":
            return 0
        return int(text) if text.isdigit() else None

    def on_exit(self, code):
        stopping = self.job.stopping
        self.job = None
        if code not in (0, None) and not stopping:
            self.set_phase("error")
        elif self.phase not in ("over", "reached"):
            self.set_phase("stopped")  # keep a game-over message on screen
        else:
            self.update_buttons()

    def update_buttons(self):
        running = self.job is not None and self.job.running
        if running:
            self.run_btn.config(text="■  Stop")
            self.run_btn.set_kind("danger")
            self.run_btn.set_enabled(self.phase != "stopping")
        else:
            self.run_btn.config(text="▶  Start AI")
            self.run_btn.set_kind("gold")
            self.run_btn.set_enabled(IS_WINDOWS)

    def on_jobs_changed(self):
        self.update_buttons()

    def draw_best(self):
        c = self.num["Highest tile"]
        c.delete("all")
        self.drawn_best = self.best
        if not self.best:
            c.create_text(0, px(17), text="-", anchor="w", font=FONT_STAT, fill=TEXT)
            return
        bg, fg = tile_colors(self.best)
        text = f"{self.best:,}"
        font = ("Segoe UI Black", 12)
        w = max(px(44), theme.tkfont.Font(font=font).measure(text) + px(22))
        round_rect(c, 0, 1, w, px(33), px(6), fill=bg, outline="")
        c.create_text(w / 2, px(17), text=text, fill=fg, font=font)

    def tick(self):
        if self.job and self.job.running:
            elapsed = time.time() - self.job.started
            self.num["Moves"].config(text=f"{self.moves:,}")
            self.num["Moves / second"].config(text=f"{self.moves / elapsed:.1f}" if self.moves else "-")
            self.num["Thinking time"].config(
                text=f"{sum(self.think_ms) / len(self.think_ms):.0f} ms" if self.think_ms else "-")
            self.num["Running for"].config(text=format_duration(elapsed))
            if self.drawn_best != self.best:
                self.draw_best()
            if self.phase == "found" and self.countdown_end:
                left = max(0, int(self.countdown_end - time.time() + 0.999))
                self.message.config(text=f"Board found. Click on the game page now. First move in {left}s."
                                    if left else "Board found. Starting...")
        self.after(250, self.tick)


class TrainPage(Page):
    title = "Train"
    subtitle = "The model improves by playing games against itself. Stop any time: progress is saved."

    def __init__(self, app):
        super().__init__(app)
        s = app.settings["train"]
        self.until_stop = tk.BooleanVar(value=s["until_stop"])
        self.games = tk.StringVar(value=str(s["games"]))
        self.corner = tk.DoubleVar(value=s["corner"])
        self.alpha = tk.StringVar(value=str(s["alpha"]))
        self.batch = tk.StringVar(value=str(s["batch"]))
        self.save_minutes = tk.StringVar(value=str(s["save_minutes"]))
        e = app.settings["eval"]
        self.eval_games = tk.StringVar(value=str(e["games"]))
        self.eval_depth = tk.StringVar(value=str(e["depth"]))
        self.job = None
        self.eval_job = None

        self.body.columnconfigure(0, weight=0, minsize=px(380))
        self.body.columnconfigure(1, weight=1)
        self.body.rowconfigure(0, weight=1)

        left = tk.Frame(self.body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))

        opts = Card(left, "Training")
        opts.pack(fill="x")
        self.model_label = WrapLabel(opts, font=FONT, bg=PANEL, fg=MUTED, anchor="w", justify="left",
)
        self.model_label.pack(fill="x")
        length = field(opts, "How long")
        ttk.Radiobutton(length, text="Until I press Stop", variable=self.until_stop, value=True
                        ).pack(anchor="w")
        row = tk.Frame(length, bg=PANEL)
        row.pack(anchor="w", pady=(2, 0))
        ttk.Radiobutton(row, text="Stop after", variable=self.until_stop, value=False).pack(side="left")
        ttk.Spinbox(row, from_=1000, to=10_000_000, increment=10000, textvariable=self.games,
                    width=10).pack(side="left", padx=6)
        tk.Label(row, text="games", font=FONT, bg=PANEL, fg=TEXT).pack(side="left")
        SliderField(field(opts, "Corner strategy strength",
                          "Moves are picked with the bottom-right corner strategy, so the model learns "
                          "the positions it will really play. 0 = off."), self.corner, 0, 300)

        adv = field(opts, "Advanced")
        grid = tk.Frame(adv, bg=PANEL)
        grid.pack(fill="x")
        for i, (label, var, lo, hi, step) in enumerate([
                ("Learning rate", self.alpha, 0.001, 1.0, 0.01),
                ("Games at once", self.batch, 16, 4096, 64),
                ("Save every (min)", self.save_minutes, 1, 120, 1)]):
            tk.Label(grid, text=label, font=FONT_SMALL, bg=PANEL, fg=MUTED).grid(row=0, column=i, sticky="w",
                                                                              padx=(0, 8))
            ttk.Spinbox(grid, from_=lo, to=hi, increment=step, textvariable=var, width=9
                        ).grid(row=1, column=i, sticky="w", padx=(0, 8))

        buttons = tk.Frame(left, bg=BG)
        buttons.pack(fill="x", pady=(14, 0))
        self.start_btn = FlatButton(buttons, "▶  Start training", self.start, kind="gold")
        self.start_btn.pack(side="left")
        self.stop_btn = FlatButton(buttons, "■  Stop and save", self.stop, kind="danger")
        self.stop_btn.pack(side="left", padx=(8, 0))

        ev = Card(left, "Evaluate", "Play test games with search (no learning) to see how strong "
                                    "the model really is. Results appear on the Stats page.")
        ev.pack(fill="x", pady=(14, 0))
        row = tk.Frame(ev, bg=PANEL)
        row.pack(fill="x", pady=(4, 0))
        tk.Label(row, text="Games", font=FONT_SMALL, bg=PANEL, fg=MUTED).grid(row=0, column=0, sticky="w")
        ttk.Spinbox(row, from_=10, to=10000, increment=50, textvariable=self.eval_games, width=8
                    ).grid(row=1, column=0, sticky="w", padx=(0, 10))
        tk.Label(row, text="Search depth", font=FONT_SMALL, bg=PANEL, fg=MUTED).grid(row=0, column=1, sticky="w")
        ttk.Spinbox(row, from_=0, to=3, increment=1, textvariable=self.eval_depth, width=6
                    ).grid(row=1, column=1, sticky="w", padx=(0, 10))
        self.eval_btn = FlatButton(row, "Run evaluation", self.start_eval, kind="primary", small=True)
        self.eval_btn.grid(row=1, column=2, sticky="w")
        self.eval_status = WrapLabel(ev, font=FONT_SMALL, bg=PANEL, fg=MUTED, anchor="w", justify="left",
)
        self.eval_status.pack(fill="x", pady=(6, 0))

        # ---- right: live progress + log
        right = tk.Frame(self.body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        live = Card(right)
        live.grid(row=0, column=0, sticky="nsew")
        head = tk.Frame(live, bg=PANEL)
        head.pack(fill="x")
        self.state_label = tk.Label(head, font=FONT_BOLD, bg=PANEL, fg=MUTED, anchor="w")
        self.state_label.pack(side="left")
        self.games_label = tk.Label(live, font=FONT_BIG, bg=PANEL, fg=TEXT, anchor="w")
        self.games_label.pack(fill="x", pady=(4, 0))
        self.progress = theme.ProgressBar(live, maximum=1000)
        self.progress.pack(fill="x", pady=(6, 2))
        self.progress_text = tk.Label(live, font=FONT_SMALL, bg=PANEL, fg=MUTED, anchor="w")
        self.progress_text.pack(fill="x")
        nums = tk.Frame(live, bg=PANEL)
        nums.pack(fill="x", pady=(10, 0))
        self.num = {}
        for i, name in enumerate(("Avg score", "Reaches 2048", "Reaches 4096", "Best tile", "Games / s")):
            nums.columnconfigure(i, weight=1)
            tk.Label(nums, text=name, font=FONT_SMALL, bg=PANEL, fg=MUTED, anchor="w").grid(row=0, column=i, sticky="w")
            lab = tk.Label(nums, text="-", font=("Segoe UI Semibold", 16), bg=PANEL, fg=TEXT, anchor="w")
            lab.grid(row=1, column=i, sticky="w")
            self.num[name] = lab
        WrapLabel(live, text="Numbers cover the most recent training games (played without search). "
                             "Full charts are on the Stats page.", font=FONT_SMALL, bg=PANEL, fg=MUTED,
                  anchor="w", justify="left").pack(fill="x", pady=(8, 0))

        self.log = LogView(right, height=10, title="Training output")
        self.log.grid(row=1, column=0, sticky="nsew", pady=(14, 0))

        self.on_model_changed()
        self.tick()

    # ---- settings

    def number(self, var, cast, name, lo, hi):
        try:
            value = cast(var.get())
        except ValueError:
            raise ValueError(f"{name} must be a number.")
        if not lo <= value <= hi:
            raise ValueError(f"{name} must be between {lo:g} and {hi:g}.")
        return value

    def read_settings(self):
        s = {"until_stop": self.until_stop.get(), "corner": round(self.corner.get()),
             "games": self.number(self.games, int, "Number of games", 1, 100_000_000),
             "alpha": self.number(self.alpha, float, "Learning rate", 1e-6, 10),
             "batch": self.number(self.batch, int, "Games at once", 1, 100_000),
             "save_minutes": self.number(self.save_minutes, float, "Save every", 0.1, 1440)}
        self.app.settings["train"] = s
        return s

    # ---- training

    def start(self):
        try:
            s = self.read_settings()
        except ValueError as e:
            messagebox.showerror(APP_NAME, str(e))
            return
        if self.external_training():
            messagebox.showwarning(APP_NAME, "This model is already being trained in another window. "
                                             "Stop that first, or train a different model.")
            return
        os.makedirs(os.path.dirname(self.app.model), exist_ok=True)
        args = ["agent2048.py", "train", "--weights", self.app.model, "--corner", str(s["corner"]),
                "--alpha", str(s["alpha"]), "--batch", str(s["batch"]),
                "--save-every", str(s["save_minutes"] * 60), "--log-every", "15"]
        if not s["until_stop"]:
            args += ["--games", str(s["games"])]
        self.log.clear()
        self.log.write(f"Training {model_name(self.app.model)}...")
        self.job = self.app.start_job("train", args, on_line=self.log.write, on_status=self.log.status,
                                      on_exit=self.on_exit)

    def stop(self):
        if self.job:
            self.job.stop(grace=120)
            self.log.write("Stopping: finishing the current step and saving the model...")

    def on_exit(self, code):
        if code not in (0, None) and not self.job.stopping:
            self.log.write(f"Training stopped with an error (exit code {code}).")
        self.job = None
        self.app.forget_new_model(self.app.model)
        self.app.notify_model_files_changed()

    def external_training(self):
        """True if another process (e.g. a terminal) is training the active model right now."""
        if self.job and self.job.running:
            return False
        s = read_json(stats_path(self.app.model))
        return bool(s) and s.get("status") in ("training", "saving") and time.time() - s.get("updated", 0) < 15

    # ---- evaluation

    def start_eval(self):
        if not os.path.exists(self.app.model):
            messagebox.showinfo(APP_NAME, "Train this model first; there's nothing to evaluate yet.")
            return
        try:
            games = self.number(self.eval_games, int, "Games", 1, 100_000)
            depth = self.number(self.eval_depth, int, "Search depth", 0, 3)
        except ValueError as e:
            messagebox.showerror(APP_NAME, str(e))
            return
        self.app.settings["eval"] = {"games": games, "depth": depth}
        corner = str(round(self.corner.get()))
        self.eval_status.config(text=f"Playing {games} test games...", fg=MUTED)
        self.eval_job = self.app.start_job(
            "eval", ["agent2048.py", "eval", "--weights", self.app.model, "--games", str(games),
                     "--depth", str(depth), "--corner", corner, "--log-every", "5"],
            on_line=self.on_eval_line, on_exit=self.on_eval_exit)

    def on_eval_line(self, line):
        line = line.strip()
        if "games finished" in line:  # periodic progress: "  12s: 40/200 games finished, ..."
            self.eval_status.config(text=line.split(":", 1)[1].strip())
        elif line:
            self.log.write("[eval] " + line)

    def on_eval_exit(self, code):
        ev = read_json(eval_path(self.app.model))
        if code == 0 and ev:
            table = {r["tile"]: r["rate"] for r in ev.get("tiles", [])}
            self.eval_status.config(fg=TEXT, text=(
                f"{ev['games']} games: avg score {ev['avg_score']:,.0f}, best tile {ev['best_tile']:,}\n"
                f"2048: {100 * table.get(2048, 0):.0f}%   4096: {100 * table.get(4096, 0):.0f}%   "
                f"8192: {100 * table.get(8192, 0):.0f}%   16384: {100 * table.get(16384, 0):.0f}%"))
        elif self.eval_job and self.eval_job.stopping:
            self.eval_status.config(text="Evaluation stopped.", fg=MUTED)
        else:
            self.eval_status.config(text="Evaluation failed - see the output.", fg=BAD)
        self.eval_job = None

    # ---- live view

    def on_model_changed(self):
        path = self.app.model
        if os.path.exists(path):
            info = model_info(path)
            text = (f"Model: {model_name(path)}  ({games_trained(path):,} games so far)\n"
                    "Training continues from where it left off.")
        else:
            text = f"Model: {model_name(path)}  (new - training starts from scratch)"
        self.model_label.config(text=text)
        self.eval_status.config(text="", fg=MUTED)

    def on_jobs_changed(self):
        training = self.job is not None and self.job.running
        evaluating = self.eval_job is not None and self.eval_job.running
        self.start_btn.set_enabled(not training)
        self.stop_btn.set_enabled(training)
        self.eval_btn.set_enabled(not evaluating)
        self.eval_btn.config(text="Running..." if evaluating else "Run evaluation")

    def tick(self):
        s = read_json(stats_path(self.app.model)) if os.path.exists(stats_path(self.app.model)) else None
        mine = self.job is not None and self.job.running
        if s is None:
            self.state_label.config(text="● Training" if mine else "Not trained yet",
                                    fg=GOOD if mine else MUTED)
            self.games_label.config(text="Starting..." if mine else "0 games trained")
            self.progress.config(value=0)
            self.progress_text.config(text="")
            for lab in self.num.values():
                lab.config(text="-")
        else:
            fresh = time.time() - s.get("updated", 0) < 15
            active = s.get("status") in ("training", "saving") and fresh
            if active:
                where = "" if mine else "  (in another window)"
                self.state_label.config(text=("● Saving" if s["status"] == "saving" else "● Training") + where,
                                        fg=GOOD)
            elif mine:
                self.state_label.config(text="● Starting...", fg=GOOD)
            else:
                self.state_label.config(text=f"■ Stopped {fmt_ago(s.get('updated'))}", fg=MUTED)
            self.games_label.config(text=f"{s.get('games_trained', 0):,} games trained")
            run, target = s.get("games_this_run", 0), s.get("target_games")
            rate = s.get("games_per_sec", 0.0)
            if target and active:
                self.progress.config(value=1000 * min(run / target, 1))
                eta = format_duration((target - run) / rate) if rate > 0 else "-"
                self.progress_text.config(text=f"This run: {run:,} of {target:,} games  ·  about {eta} left")
            else:
                self.progress.config(value=0)
                self.progress_text.config(text=f"This run: {run:,} games  ·  {format_duration(s.get('elapsed', 0))}"
                                          if active else "")
            table = {r["tile"]: r["rate"] for r in s.get("tiles", [])}
            self.num["Avg score"].config(text=f"{s.get('avg_score', 0):,.0f}")
            self.num["Reaches 2048"].config(text=f"{100 * table.get(2048, 0):.0f}%" if table else "-")
            self.num["Reaches 4096"].config(text=f"{100 * table.get(4096, 0):.0f}%" if table else "-")
            self.num["Best tile"].config(text=f"{s.get('best_tile_ever', 0):,}")
            self.num["Games / s"].config(text=f"{rate:.1f}" if active else "-")
        self.after(1000, self.tick)


class ModelsPage(Page):
    title = "Models"
    subtitle = "Each model is a weights file. Pick the one to train and play with, or add new ones."

    def __init__(self, app):
        super().__init__(app)
        self.busy = False
        self.body.columnconfigure(0, weight=1)
        self.body.columnconfigure(1, weight=0, minsize=px(330))
        self.body.rowconfigure(0, weight=1)

        left = Card(self.body, padx=0, pady=0)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        cols = ("name", "games", "score", "size", "modified")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", selectmode="browse",
                                 style="Models.Treeview")
        for col, text, width, anchor in (("name", "Model", 190, "w"), ("games", "Games trained", 100, "e"),
                                         ("score", "Eval avg score", 100, "e"), ("size", "Size", 75, "e"),
                                         ("modified", "Last changed", 110, "e")):
            self.tree.heading(col, text=text, anchor=anchor)
            self.tree.column(col, width=px(width), anchor=anchor, stretch=col == "name")
        self.tree.tag_configure("active", font=FONT_BOLD)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", lambda e: self.show_details())
        self.tree.bind("<Double-1>", lambda e: self.use_selected())

        right = tk.Frame(self.body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        self.details = Card(right, "Details")
        self.details.pack(fill="x")
        self.detail_text = WrapLabel(self.details, font=FONT, bg=PANEL, fg=TEXT, anchor="w", justify="left",
)
        self.detail_text.pack(fill="x", pady=(4, 10))
        self.use_btn = FlatButton(self.details, "Use this model", self.use_selected, kind="gold")
        self.use_btn.pack(anchor="w")

        manage = Card(right, "Manage")
        manage.pack(fill="x", pady=(14, 0))
        self.buttons = []
        for text, cmd, needs_sel in (("New model...", self.new_model, False),
                                     ("Duplicate...", self.duplicate, True),
                                     ("Rename...", self.rename, True),
                                     ("Delete...", self.delete, True),
                                     ("Import...", self.import_model, False),
                                     ("Export for sharing...", self.export_model, True),
                                     ("Open models folder", self.open_folder, False)):
            b = FlatButton(manage, text, cmd, kind="secondary", small=True, anchor="w")
            b.pack(fill="x", pady=2)
            self.buttons.append((b, needs_sel))
        if PRETRAINED_URL:
            b = FlatButton(manage, "Download pretrained model", self.download_pretrained, kind="gold", small=True,
                           anchor="w")
            b.pack(fill="x", pady=(8, 2))
            self.buttons.append((b, False))
        self.busy_label = WrapLabel(manage, font=FONT_SMALL, bg=PANEL, fg=MUTED, anchor="w", justify="left",
)
        self.busy_label.pack(fill="x", pady=(6, 0))
        WrapLabel(right, text="Export makes a compact .npz copy (about a tenth of the size) that "
                             "you can share or attach to a GitHub release; Import accepts it.",
                 font=FONT_SMALL, bg=BG, fg=MUTED, justify="left", anchor="w"
                 ).pack(fill="x", pady=(10, 0))

    # ---- list

    def on_show(self):
        self.refresh()

    def on_model_changed(self):
        self.refresh()

    def refresh(self):
        selected = self.selected()
        self.tree.delete(*self.tree.get_children())
        for path in find_models(self.app.settings["new_models"] + [self.app.model]):
            exists = os.path.exists(path)
            info = model_info(path) if exists else {}
            ev = read_json(eval_path(path)) if exists else None
            name = model_name(path) + ("   ✓ in use" if path == self.app.model else "")
            values = (name, f"{games_trained(path):,}" if exists else "new",
                      f"{ev['avg_score']:,.0f}" if ev else "-",
                      fmt_size(os.path.getsize(path)) if exists else "-",
                      time.strftime("%b %d, %H:%M", time.localtime(os.path.getmtime(path))) if exists else "-")
            self.tree.insert("", "end", iid=path, values=values,
                             tags=("active",) if path == self.app.model else ())
        target = selected if selected and self.tree.exists(selected) else self.app.model
        if self.tree.exists(target):
            self.tree.selection_set(target)
        self.show_details()

    def selected(self):
        sel = self.tree.selection()
        return sel[0] if sel else None

    def show_details(self):
        path = self.selected()
        for b, needs_sel in self.buttons:
            b.set_enabled(not self.busy and (path is not None or not needs_sel))
        if not path:
            self.detail_text.config(text="Select a model.")
            self.use_btn.set_enabled(False)
            return
        exists = os.path.exists(path)
        lines = [model_name(path), ""]
        if exists:
            info = model_info(path)
            lines.append(f"Games trained: {games_trained(path):,}")
            if "corner" in info:
                lines.append(f"Corner strategy in training: {info['corner']:g}")
            if "alpha" in info:
                lines.append(f"Learning rate: {info['alpha']:g}")
            ev = read_json(eval_path(path))
            if ev:
                lines.append(f"Last evaluation: avg score {ev['avg_score']:,.0f}, best tile {ev['best_tile']:,}")
            lines.append(f"File: {rel_path(path)}")
        else:
            lines.append("Not trained yet. Select it and open Train to start it from scratch.")
        self.detail_text.config(text="\n".join(lines))
        self.use_btn.set_enabled(path != self.app.model and not self.busy)

    def in_use_by_job(self, path):
        return path == self.app.model and self.app.any_job_running()

    # ---- actions

    def use_selected(self):
        path = self.selected()
        if not path or path == self.app.model:
            return
        if self.app.any_job_running():
            messagebox.showinfo(APP_NAME, "Stop training, evaluation and play before switching models.")
            return
        self.app.set_model(path)

    def ask_name(self, title, initial=""):
        while True:
            name = simpledialog.askstring(APP_NAME, title, initialvalue=initial, parent=self)
            if name is None:
                return None
            name = name.strip()
            if not valid_model_name(name):
                messagebox.showerror(APP_NAME, "Use letters, numbers, spaces, dashes, dots or underscores.")
                continue
            path = os.path.join(MODELS_DIR, name + ".npy")
            if os.path.exists(path) or path in self.app.settings["new_models"]:
                messagebox.showerror(APP_NAME, f"A model called '{name}' already exists.")
                continue
            return path

    def new_model(self):
        path = self.ask_name("Name for the new model:")
        if not path:
            return
        self.app.settings["new_models"].append(path)
        if not self.app.any_job_running():
            self.app.set_model(path)
            if messagebox.askyesno(APP_NAME, f"'{model_name(path)}' is ready. Go to Train to start it?"):
                self.app.show("train")
        else:
            self.refresh()

    def duplicate(self):
        src = self.selected()
        if not src or not os.path.exists(src):
            return
        dst = self.ask_name("Name for the copy:", model_name(src) + " copy")
        if dst:
            self.run_busy(f"Copying {model_name(src)}...", lambda: self.copy_model(src, dst))

    @staticmethod
    def copy_model(src, dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        for a, b in zip(sidecar_files(src), sidecar_files(dst)):
            if os.path.exists(a):
                shutil.copyfile(a, b)

    def rename(self):
        src = self.selected()
        if not src:
            return
        if self.in_use_by_job(src):
            messagebox.showinfo(APP_NAME, "Stop training, evaluation and play before renaming this model.")
            return
        dst = self.ask_name("New name:", model_name(src))
        if not dst:
            return
        if os.path.exists(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.replace(src, dst)
            for a, b in zip(sidecar_files(src), sidecar_files(dst)):
                if os.path.exists(a):
                    os.replace(a, b)
        new = self.app.settings["new_models"]
        if src in new:
            new[new.index(src)] = dst
        if src == self.app.model:
            self.app.set_model(dst)
        self.refresh()
        self.tree.selection_set(dst)

    def delete(self):
        path = self.selected()
        if not path:
            return
        if self.in_use_by_job(path):
            messagebox.showinfo(APP_NAME, "Stop training, evaluation and play before deleting this model.")
            return
        info = model_info(path) if os.path.exists(path) else {}
        if not messagebox.askyesno(APP_NAME, f"Delete '{model_name(path)}' "
                                             f"({games_trained(path):,} games of training)?\n\n"
                                             "This can't be undone.", icon="warning"):
            return
        for f in [path] + sidecar_files(path):
            if os.path.exists(f):
                os.remove(f)
        if path in self.app.settings["new_models"]:
            self.app.settings["new_models"].remove(path)
        if path == self.app.model:
            remaining = [p for p in find_models() if os.path.exists(p)]
            self.app.set_model(remaining[0] if remaining else DEFAULT_WEIGHTS)
        self.refresh()

    def import_model(self):
        src = filedialog.askopenfilename(parent=self, title="Import a model",
                                         filetypes=[("2048 AI models", "*.npz *.npy"), ("All files", "*.*")])
        if not src:
            return
        dst = self.ask_name("Name for the imported model:", model_name(src))
        if dst:
            self.run_busy(f"Importing {os.path.basename(src)}...", lambda: self.import_file(src, dst))

    @staticmethod
    def import_file(src, dst):
        import numpy as np
        from agent2048 import TABLE_SIZE, TUPLES
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        expected = len(TUPLES) * TABLE_SIZE
        if src.lower().endswith(".npz"):
            weights, info = load_compact(src)
            if weights.size != expected:
                raise ValueError("This file is for a different network layout.")
            np.save(dst, weights)
            write_json(os.path.splitext(dst)[0] + ".json", info)
        else:
            weights = np.load(src, mmap_mode="r")
            if weights.size != expected or weights.dtype != np.float32:
                raise ValueError("This file isn't a 2048 AI weights file.")
            shutil.copyfile(src, dst)
            for a, b in zip(sidecar_files(src), sidecar_files(dst)):
                if os.path.exists(a):
                    shutil.copyfile(a, b)

    def export_model(self):
        src = self.selected()
        if not src or not os.path.exists(src):
            return
        dst = filedialog.asksaveasfilename(parent=self, title="Export model", defaultextension=".npz",
                                           initialfile=model_name(src) + ".npz",
                                           filetypes=[("Compact 2048 AI model", "*.npz")])
        if not dst:
            return

        def work():
            import numpy as np
            info = model_info(src)
            save_compact(np.load(src, mmap_mode="r"), dst, info)
            return f"Exported to {dst} ({fmt_size(os.path.getsize(dst))})."
        self.run_busy(f"Exporting {model_name(src)}...", work)

    def open_folder(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        if IS_WINDOWS:
            os.startfile(MODELS_DIR)
        else:
            webbrowser.open("file://" + MODELS_DIR)

    def download_pretrained(self):
        if not PRETRAINED_URL:
            return
        dst = os.path.join(MODELS_DIR, "pretrained.npy")
        if os.path.exists(dst) and not messagebox.askyesno(APP_NAME, "Replace the downloaded pretrained model?"):
            return

        def work():
            tmp = os.path.join(tempfile.gettempdir(), "ai2048_pretrained.npz")
            with urllib.request.urlopen(PRETRAINED_URL) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    msg = f"Downloading... {fmt_size(done)}" + (f" of {fmt_size(total)}" if total else "")
                    self.after(0, lambda m=msg: self.busy_label.config(text=m))
            self.import_file(tmp, dst)
            os.remove(tmp)
            self.after(0, lambda: self.app.set_model(dst))
            return "Pretrained model downloaded and selected."
        self.app.show("models")
        self.run_busy("Downloading the pretrained model...", work)

    def run_busy(self, message, work):
        """Run slow file work on a thread, keeping the window responsive."""
        self.busy = True
        self.busy_label.config(text=message, fg=MUTED)
        self.show_details()

        def done(result, error):
            self.busy = False
            if error:
                self.busy_label.config(text=f"Failed: {error}", fg=BAD)
            else:
                self.busy_label.config(text=result or "Done.", fg=GOOD)
            self.refresh()

        def run():
            try:
                result = work()
            except Exception as e:  # shown to the user
                self.after(0, lambda e=e: done(None, e))
            else:
                self.after(0, lambda: done(result, None))
        threading.Thread(target=run, daemon=True).start()


class StatsPage(Page):
    title = "Stats"
    subtitle = "Live training progress and the latest evaluation for the active model."

    def __init__(self, app):
        super().__init__(app)
        # Scrolls in a short window, so the charts always get room to draw.
        scroll = theme.ScrollColumn(self.body, width=px(600))
        scroll.pack(fill="both", expand=True)
        self.dash = StatsDashboard(scroll.inner, app.model, title=model_name(app.model))
        self.dash.config(padx=0, pady=0)
        self.dash.pack(fill="both", expand=True)

    def on_model_changed(self):
        self.dash.set_weights(self.app.model)
        self.dash.title_label.config(text=model_name(self.app.model))


# ---------------------------------------------------------------- main window

class App(tk.Tk):
    NAV = [("home", "Home", "⌂"), ("play", "Play", "▶"), ("train", "Train", "↻"),
           ("models", "Models", "▦"), ("stats", "Stats", "▤")]

    def __init__(self):
        super().__init__()
        theme.set_scale(max(1.0, self.winfo_fpixels("1i") / 96))
        self.settings = load_settings()
        self.model = self.settings["model"]
        self.events = queue.Queue()
        self.jobs = {}

        self.title(APP_NAME)
        self.configure(bg=BG)
        width = min(px(1220), self.winfo_screenwidth() - 40)
        height = min(px(820), self.winfo_screenheight() - 80)
        self.geometry(f"{width}x{height}")
        self.minsize(min(px(1080), width), min(px(760), height))
        self.icon = make_icon()
        self.iconphoto(True, self.icon)
        self.setup_style()

        # ---- sidebar
        side = tk.Frame(self, bg=SIDEBAR, width=px(224))
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        brand = tk.Frame(side, bg=SIDEBAR)
        brand.pack(fill="x", padx=px(20), pady=(px(24), px(28)))
        logo(brand, SIDEBAR, px(38)).pack(side="left")
        names = tk.Frame(brand, bg=SIDEBAR)
        names.pack(side="left", padx=(px(12), 0))
        tk.Label(names, text=APP_NAME, font=("Segoe UI Black", 14), bg=SIDEBAR, fg=SIDEBAR_TEXT,
                 anchor="w").pack(fill="x")
        tk.Label(names, text="Self-taught player", font=FONT_SMALL, bg=SIDEBAR, fg=SIDEBAR_MUTED,
                 anchor="w").pack(fill="x")

        tk.Label(side, text="MENU", font=FONT_CAPS, bg=SIDEBAR, fg=SIDEBAR_MUTED, anchor="w"
                 ).pack(fill="x", padx=px(24), pady=(0, px(6)))
        self.nav_items = {}
        self.hovered = None
        for key, label, icon in self.NAV:
            row = tk.Frame(side, bg=SIDEBAR, cursor="hand2")
            row.pack(fill="x", padx=px(12), pady=1)
            bar = tk.Frame(row, bg=SIDEBAR, width=px(3))
            bar.pack(side="left", fill="y")
            ic = tk.Label(row, text=icon, font=("Segoe UI Symbol", 11), bg=SIDEBAR, fg=SIDEBAR_MUTED,
                          width=2, cursor="hand2")
            ic.pack(side="left", padx=(px(10), px(6)), pady=px(9))
            tx = tk.Label(row, text=label, font=FONT_BOLD, bg=SIDEBAR, fg=SIDEBAR_MUTED, anchor="w",
                          cursor="hand2")
            tx.pack(side="left", fill="x", expand=True)
            for w in (row, ic, tx):
                w.bind("<Button-1>", lambda e, k=key: self.show(k))
                w.bind("<Enter>", lambda e, k=key: self.nav_hover(k, True))
                w.bind("<Leave>", lambda e, k=key: self.nav_hover(k, False))
            self.nav_items[key] = (row, bar, ic, tx)

        foot = tk.Frame(side, bg=SIDEBAR)
        foot.pack(side="bottom", fill="x", padx=px(12), pady=px(16))
        self.job_label = tk.Label(foot, font=FONT_SMALL_BOLD, bg=SIDEBAR, fg=BRAND, anchor="w", justify="left")
        self.job_label.pack(fill="x", padx=px(12))
        model_box = tk.Frame(foot, bg=SIDEBAR_ACTIVE, padx=px(12), pady=px(10), cursor="hand2")
        model_box.pack(fill="x", pady=(px(10), 0))
        cap = tk.Label(model_box, text="ACTIVE MODEL", font=FONT_CAPS, bg=SIDEBAR_ACTIVE, fg=SIDEBAR_MUTED,
                       anchor="w", cursor="hand2")
        cap.pack(fill="x")
        self.model_label = tk.Label(model_box, font=FONT_BOLD, bg=SIDEBAR_ACTIVE, fg=SIDEBAR_TEXT, anchor="w",
                                    cursor="hand2")
        self.model_label.pack(fill="x")
        self.model_games = tk.Label(model_box, font=FONT_SMALL, bg=SIDEBAR_ACTIVE, fg=SIDEBAR_MUTED, anchor="w",
                                    cursor="hand2")
        self.model_games.pack(fill="x")
        for w in (model_box, cap, self.model_label, self.model_games):
            w.bind("<Button-1>", lambda e: self.show("models"))

        # ---- pages
        self.content = tk.Frame(self, bg=BG)
        self.content.pack(side="left", fill="both", expand=True)
        self.pages = {"home": HomePage(self), "play": PlayPage(self), "train": TrainPage(self),
                      "models": ModelsPage(self), "stats": StatsPage(self)}
        self.current = None
        self.update_model_label()
        self.jobs_changed()
        self.show("home")
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.after(50, self.pump_events)

    def setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", font=FONT, background=PANEL, foreground=TEXT, focuscolor=PANEL)
        style.configure("TRadiobutton", background=PANEL, foreground=TEXT, indicatorbackground=PANEL,
                        indicatorforeground=BRAND_DEEP, indicatormargin=(0, 0, px(6), 0), padding=(0, 2))
        style.map("TRadiobutton", background=[("active", PANEL)],
                  indicatorbackground=[("selected", PANEL), ("active", SUBTLE)])
        field_opts = dict(fieldbackground=PANEL, background=PANEL, bordercolor=BORDER_STRONG,
                          lightcolor=PANEL, darkcolor=PANEL, arrowcolor=TEXT_2, padding=(px(8), px(5)),
                          selectbackground=BRAND_SOFT, selectforeground=TEXT)
        style.configure("TSpinbox", **field_opts)
        style.configure("TCombobox", **field_opts)
        for name in ("TSpinbox", "TCombobox"):
            style.map(name, bordercolor=[("focus", BRAND_DEEP), ("hover", MUTED)],
                      fieldbackground=[("readonly", PANEL)], background=[("active", SUBTLE)],
                      lightcolor=[("focus", PANEL)], darkcolor=[("focus", PANEL)])
        self.option_add("*TCombobox*Listbox.background", PANEL)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", BRAND_SOFT)
        self.option_add("*TCombobox*Listbox.selectForeground", TEXT)
        self.option_add("*TCombobox*Listbox.font", FONT)
        self.option_add("*TCombobox*Listbox.relief", "flat")
        style.configure("Models.Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                        rowheight=px(36), bordercolor=PANEL, borderwidth=0, lightcolor=PANEL, darkcolor=PANEL)
        style.configure("Models.Treeview.Heading", background=PANEL, foreground=MUTED, relief="flat",
                        font=FONT_CAPS, padding=(px(8), px(8)), bordercolor=BORDER, lightcolor=PANEL,
                        darkcolor=BORDER)
        style.map("Models.Treeview.Heading", background=[("active", SUBTLE)])
        style.map("Models.Treeview", background=[("selected", BRAND_SOFT)], foreground=[("selected", TEXT)])
        style.layout("Models.Treeview", [("Models.Treeview.treearea", {"sticky": "nswe"})])

    # ---- navigation

    def show(self, key):
        if self.current:
            self.pages[self.current].pack_forget()
        self.current = key
        self.pages[key].pack(fill="both", expand=True)
        self.paint_nav()
        self.pages[key].on_show()

    def nav_hover(self, key, inside):
        self.hovered = key if inside else (None if self.hovered == key else self.hovered)
        self.paint_nav()

    def paint_nav(self):
        for k, (row, bar, ic, tx) in self.nav_items.items():
            active, hover = k == self.current, k == self.hovered
            bg = SIDEBAR_ACTIVE if active else SIDEBAR_HOVER if hover else SIDEBAR
            fg = SIDEBAR_TEXT if active or hover else SIDEBAR_MUTED
            row.config(bg=bg)
            bar.config(bg=BRAND if active else bg)
            ic.config(bg=bg, fg=BRAND if active else fg)
            tx.config(bg=bg, fg=fg)

    # ---- models

    def set_model(self, path):
        self.model = os.path.normpath(path)
        self.settings["model"] = self.model
        save_settings(self.settings)
        self.update_model_label()
        for page in self.pages.values():
            page.on_model_changed()

    def forget_new_model(self, path):
        if path in self.settings["new_models"] and os.path.exists(path):
            self.settings["new_models"].remove(path)

    def notify_model_files_changed(self):
        for page in self.pages.values():
            page.on_model_changed()

    def update_model_label(self):
        self.model_label.config(text=model_name(self.model))
        games = games_trained(self.model) if os.path.exists(self.model) else 0
        self.model_games.config(text=f"{games:,} games trained" if games else "Not trained yet")

    # ---- jobs

    def start_job(self, name, args, on_line=None, on_status=None, on_exit=None):
        save_settings(self.settings)
        try:
            job = Job(self, name, args, on_line, on_status, on_exit)
        except OSError as e:
            messagebox.showerror(APP_NAME, f"Couldn't start {args[0]}: {e}")
            return None
        self.jobs[name] = job
        # Deferred: the page that called this only stores the job once we return,
        # and its buttons need to see it (otherwise Stop stays disabled).
        self.after_idle(self.jobs_changed)
        return job

    def any_job_running(self):
        return any(j.running for j in self.jobs.values())

    def jobs_changed(self):
        names = {"train": "Training", "eval": "Evaluating", "play": "AI playing"}
        running = [names[n] for n, j in self.jobs.items() if j.running]
        self.job_label.config(text="\n".join("\u25cf  " + r for r in running))
        for page in self.pages.values():
            page.on_jobs_changed()

    def pump_events(self):
        try:
            while True:
                job, kind, data = self.events.get_nowait()
                if kind == "line" and job.on_line:
                    job.on_line(data)
                elif kind == "status" and job.on_status:
                    job.on_status(data)
                elif kind == "exit":
                    job.cleanup()
                    if job.on_exit:
                        job.on_exit(data)
                    self.jobs_changed()
        except queue.Empty:
            pass
        self.after(50, self.pump_events)

    def on_close(self):
        running = [j for j in self.jobs.values() if j.running]
        if running:
            if not messagebox.askyesno(APP_NAME, "Training or play is still running. Stop it and quit?\n\n"
                                                 "Training saves before it stops, which can take a few seconds."):
                return
            for job in running:
                job.stop(grace=90)
            self.title(f"{APP_NAME} - stopping...")
            self.wait_then_close(time.time() + 95)
            return
        self.close()

    def wait_then_close(self, deadline):
        if any(j.running for j in self.jobs.values()) and time.time() < deadline:
            self.after(200, lambda: self.wait_then_close(deadline))
        else:
            for job in self.jobs.values():
                job.kill()
                job.cleanup()
            self.close()

    def close(self):
        save_settings(self.settings)
        self.destroy()


def main():
    # The packaged exe runs its own helper scripts: "2048 AI.exe --run script.py ...".
    if len(sys.argv) > 2 and sys.argv[1] == "--run":
        run_script(sys.argv[2:])
        return
    if IS_WINDOWS:
        try:  # crisp text on high-DPI screens
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
    App().mainloop()


if __name__ == "__main__":
    main()
