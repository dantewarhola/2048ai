"""Live training dashboard for the 2048 agent.

Reads the stats file that `python agent2048.py train` writes every second
(agent_weights.stats.json) and the results of the latest
`python agent2048.py eval` (agent_weights.eval.json), and keeps a small window
up to date. It only reads files, so it can be opened and closed at any time
without affecting training.

Usage:
    python stats_viewer.py
    python stats_viewer.py --weights other_weights.npy
"""

import argparse
import os
import time
import tkinter as tk
from tkinter import ttk

from agent2048 import DEFAULT_WEIGHTS, eval_path, format_duration, read_json, stats_path

REFRESH_MS = 1000
STALE_SECONDS = 15  # no stats update for this long while "training" = trainer not running

from theme import ACCENT, BAD, BG, GOOD, GRID, MUTED, PANEL, TEXT, Card, ProgressBar

# Tile colors as on 2048.org (text colour, background).
TILE_STYLE = {
    256: ("#ffffff", "#edcc61"), 512: ("#ffffff", "#edc850"), 1024: ("#ffffff", "#edc53f"),
    2048: ("#ffffff", "#edc22e"), 4096: ("#ffffff", "#3c3a32"), 8192: ("#ffffff", "#3c3a32"),
    16384: ("#ffffff", "#3c3a32"), 32768: ("#ffffff", "#3c3a32"),
}
RATE_SERIES = [("2048", "#e0b020"), ("4096", "#f65e3b"), ("8192", "#3c3a32")]

FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_BIG = ("Segoe UI", 20, "bold")


def fmt_count(n):
    if n is None:
        return "-"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 10_000:
        return f"{n / 1000:.0f}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return f"{n:,}"


def fmt_ago(ts):
    if not ts:
        return "never"
    return format_duration(max(0, time.time() - ts)) + " ago"


class Panel(Card):
    def __init__(self, parent, title):
        super().__init__(parent, padx=12, pady=8)
        tk.Label(self, text=title, font=FONT_BOLD, bg=PANEL, fg=TEXT, anchor="w").pack(fill="x", pady=(0, 4))


class StatsDashboard(tk.Frame):
    """The whole dashboard as one widget, so the desktop app (app.py) can embed it."""

    def __init__(self, parent, weights, title="2048 Agent Training"):
        super().__init__(parent, bg=BG, padx=14, pady=12)
        self.set_weights(weights)

        outer = self

        # ---- header
        head = tk.Frame(outer, bg=BG)
        head.pack(fill="x")
        self.header = head
        self.title_label = tk.Label(head, text=title, font=FONT_TITLE, bg=BG, fg=TEXT)
        self.title_label.pack(side="left")
        self.status = tk.Label(head, text="", font=FONT_BOLD, bg=BG, fg=MUTED)
        self.status.pack(side="right", padx=12)

        # ---- progress
        prog = tk.Frame(outer, bg=BG, pady=8)
        prog.pack(fill="x")
        self.games_label = tk.Label(prog, text="", font=FONT_BIG, bg=BG, fg=TEXT, anchor="w")
        self.games_label.pack(fill="x")
        self.progress = ProgressBar(prog, maximum=1000)
        self.progress.pack(fill="x", pady=(4, 2))
        self.progress_text = tk.Label(prog, text="", font=FONT_SMALL, bg=BG, fg=MUTED, anchor="w")
        self.progress_text.pack(fill="x")

        # ---- middle: tile confidence table + numbers
        mid = tk.Frame(outer, bg=BG)
        mid.pack(fill="x", pady=(4, 8))
        mid.columnconfigure(0, weight=3)
        mid.columnconfigure(1, weight=2)

        tiles = Panel(mid, "Highest tile reached (recent training games, no search)")
        tiles.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.reliable = tk.Label(tiles, text="", font=FONT, bg=PANEL, fg=TEXT, anchor="w", justify="left")
        self.reliable.pack(fill="x", pady=(2, 4))
        self.tile_canvas = tk.Canvas(tiles, height=8 * 26 + 6, bg=PANEL, highlightthickness=0)
        self.tile_canvas.pack(fill="x")
        self.tile_canvas.bind("<Configure>", lambda e: self.draw_tiles())

        nums = Panel(mid, "Numbers")
        nums.grid(row=0, column=1, sticky="nsew")
        self.num_labels = {}
        grid = tk.Frame(nums, bg=PANEL)
        grid.pack(fill="x", pady=(4, 0))
        rows = ["Avg score", "Best score (recent)", "Best score ever", "Best tile ever",
                "Avg moves / game", "Games / sec", "Moves / sec", "Run time",
                "Last checkpoint", "Learning rate", "Games at once", "Recent window"]
        for i, name in enumerate(rows):
            tk.Label(grid, text=name, font=FONT_SMALL, bg=PANEL, fg=MUTED, anchor="w"
                     ).grid(row=i, column=0, sticky="w", pady=1)
            lab = tk.Label(grid, text="-", font=FONT_BOLD, bg=PANEL, fg=TEXT, anchor="e")
            lab.grid(row=i, column=1, sticky="e", pady=1)
            self.num_labels[name] = lab
        grid.columnconfigure(1, weight=1)

        # ---- eval (packed at the bottom first so the charts never push it off-screen)
        ev = Panel(outer, "Latest evaluation (python agent2048.py eval, with search)")
        ev.pack(side="bottom", fill="x", pady=(8, 0))
        self.eval_label = tk.Label(ev, text="", font=FONT_SMALL, bg=PANEL, fg=TEXT,
                                   anchor="w", justify="left")
        self.eval_label.pack(fill="x")

        # ---- charts
        charts = tk.Frame(outer, bg=BG)
        charts.pack(fill="both", expand=True)
        charts.columnconfigure(0, weight=1)
        charts.columnconfigure(1, weight=1)
        charts.rowconfigure(0, weight=1)
        p1 = Panel(charts, "Average score")
        p1.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        p2 = Panel(charts, "% of games reaching tile")
        p2.grid(row=0, column=1, sticky="nsew")
        self.score_chart = tk.Canvas(p1, bg=PANEL, highlightthickness=0, height=190)
        self.score_chart.pack(fill="both", expand=True)
        self.rate_chart = tk.Canvas(p2, bg=PANEL, highlightthickness=0, height=190)
        self.rate_chart.pack(fill="both", expand=True)
        self.score_chart.bind("<Configure>", lambda e: self.draw_charts())
        self.rate_chart.bind("<Configure>", lambda e: self.draw_charts())

        self.refresh()

    def set_weights(self, weights):
        """Show the stats of another weights file."""
        self.stats_file = stats_path(weights)
        self.eval_file = eval_path(weights)
        self.stats = None
        self.eval = None
        self.stats_mtime = None
        self.eval_mtime = None
        if hasattr(self, "tile_canvas"):
            self.draw_tiles()
            self.draw_charts()
            self.update_text()

    # ------------------------------------------------------------ data

    def load(self):
        for attr, path in (("stats", self.stats_file), ("eval", self.eval_file)):
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime != getattr(self, attr + "_mtime"):
                data = read_json(path)
                if data is not None:  # keep the old data if the file was mid-write
                    setattr(self, attr, data)
                    setattr(self, attr + "_mtime", mtime)
                    if attr == "stats":
                        self.draw_tiles()
                        self.draw_charts()

    def refresh(self):
        self.load()
        self.update_text()
        self.after(REFRESH_MS, self.refresh)

    # ------------------------------------------------------------ text

    def update_text(self):
        s = self.stats
        if s is None:
            self.status.config(text="Waiting for training to start", fg=MUTED)
            self.games_label.config(text="No stats yet")
            self.progress_text.config(text=f"Looking for {self.stats_file}")
            self.update_eval()
            return

        age = time.time() - s.get("updated", 0)
        state = s.get("status")
        if state in ("training", "saving") and age < STALE_SECONDS:
            text = "● Training" if state == "training" else "● Saving checkpoint"
            self.status.config(text=text, fg=GOOD)
        elif state == "stopped":
            self.status.config(text=f"■ Stopped ({fmt_ago(s.get('updated'))})", fg=MUTED)
        else:
            self.status.config(text=f"▲ No update for {format_duration(age)} (trainer closed?)", fg=BAD)

        total = s.get("games_trained", 0)
        run = s.get("games_this_run", 0)
        target = s.get("target_games")
        rate = s.get("games_per_sec", 0.0)
        self.games_label.config(text=f"{total:,} games trained")
        if target:
            frac = min(run / target, 1.0)
            self.progress.config(value=frac * 1000)
            eta = format_duration((target - run) / rate) if rate > 0 and run < target else "-"
            self.progress_text.config(
                text=f"This run: {run:,} / {target:,} games ({100 * frac:.1f}%)  ·  ETA {eta}")
        else:
            self.progress.config(value=0)
            self.progress_text.config(text=f"This run: {run:,} games  ·  no target set (runs until Ctrl+C)")

        n = self.num_labels
        n["Avg score"].config(text=f"{s.get('avg_score', 0):,.0f}")
        n["Best score (recent)"].config(text=f"{s.get('max_score_recent', 0):,}")
        n["Best score ever"].config(text=f"{s.get('best_score_ever', 0):,}")
        n["Best tile ever"].config(text=f"{s.get('best_tile_ever', 0):,}")
        n["Avg moves / game"].config(text=f"{s.get('avg_moves', 0):,.0f}")
        n["Games / sec"].config(text=f"{rate:,.1f}")
        n["Moves / sec"].config(text=f"{s.get('moves_per_sec', 0):,.0f}")
        n["Run time"].config(text=format_duration(s.get("elapsed", 0)))
        n["Last checkpoint"].config(text=fmt_ago(s.get("last_save")))
        n["Learning rate"].config(text=f"{s.get('alpha', 0):g}")
        n["Games at once"].config(text=f"{s.get('batch', 0):,}")
        n["Recent window"].config(text=f"last {s.get('window', 0):,} games")

        table = s.get("tiles") or []
        rel = s.get("reliable_tile")
        top = max((r for r in table if r["count"] > 0), key=lambda r: r["tile"], default=None)
        lines = []
        if rel:
            lines.append(f"Reliable tile: {rel:,}  (reached in ≥95% of games, 95% confidence)")
        else:
            lines.append("Reliable tile: none yet  (no tile reached in ≥95% of games with 95% confidence)")
        if top:
            lines.append(f"Highest reached recently: {top['tile']:,} in {100 * top['rate']:.1f}% of games")
        self.reliable.config(text="\n".join(lines))
        self.update_eval()

    def update_eval(self):
        e = self.eval
        if not e:
            self.eval_label.config(text="No evaluation yet. Run one from the app's Train page, or: python agent2048.py eval")
            return
        parts = [f"{fmt_ago(e.get('time'))}  ·  {e.get('games', 0):,} games, depth {e.get('depth')}"
                 f"  ·  network had {fmt_count(e.get('games_trained'))} training games"
                 f"  ·  avg score {e.get('avg_score', 0):,.0f}  ·  best tile {e.get('best_tile', 0):,}"]
        rel = e.get("reliable_tile")
        parts.append(f"Reliable tile (≥95% with 95% confidence): {rel:,}" if rel else
                     "Reliable tile (≥95% with 95% confidence): none")
        rows = [r for r in e.get("tiles", []) if r["tile"] >= 1024]
        parts.append("   ".join(f"{r['tile']}: {100 * r['rate']:.1f}% "
                                f"({100 * r['lo']:.1f}–{100 * r['hi']:.1f})" for r in rows))
        self.eval_label.config(text="\n".join(parts))

    # ------------------------------------------------------------ drawing

    def draw_tiles(self):
        c = self.tile_canvas
        c.delete("all")
        table = (self.stats or {}).get("tiles") or []
        w = max(c.winfo_width(), 300)
        if not table:
            c.create_text(8, 20, text="Waiting for finished games...", anchor="w", font=FONT_SMALL, fill=MUTED)
            return
        bar_x0, bar_x1 = 70, w - 150
        for i, row in enumerate(table):
            y = 4 + i * 26
            fg, bg = TILE_STYLE.get(row["tile"], ("#ffffff", ACCENT))
            c.create_rectangle(0, y, 58, y + 21, fill=bg, outline="")
            c.create_text(29, y + 11, text=f"{row['tile']:,}", fill=fg, font=FONT_BOLD)
            # bar = observed rate, whisker = 95% confidence interval
            c.create_rectangle(bar_x0, y + 5, bar_x1, y + 16, fill=GRID, outline="")
            span = bar_x1 - bar_x0
            c.create_rectangle(bar_x0, y + 5, bar_x0 + span * row["rate"], y + 16, fill=bg, outline="")
            lo_x, hi_x = bar_x0 + span * row["lo"], bar_x0 + span * row["hi"]
            c.create_line(lo_x, y + 10, hi_x, y + 10, fill=TEXT, width=1)
            c.create_line(lo_x, y + 6, lo_x, y + 15, fill=TEXT)
            c.create_line(hi_x, y + 6, hi_x, y + 15, fill=TEXT)
            c.create_text(bar_x1 + 8, y + 11, anchor="w", font=FONT_BOLD, fill=TEXT,
                          text=f"{100 * row['rate']:5.1f}%")
            c.create_text(bar_x1 + 62, y + 11, anchor="w", font=FONT_SMALL, fill=MUTED,
                          text=f"{100 * row['lo']:.1f}–{100 * row['hi']:.1f}%")
        # 95% marker
        x95 = bar_x0 + (bar_x1 - bar_x0) * 0.95
        c.create_line(x95, 0, x95, 4 + len(table) * 26, fill=GOOD, dash=(2, 3))

    def draw_charts(self):
        history = (self.stats or {}).get("history") or []
        xs = [h["games"] for h in history]
        draw_chart(self.score_chart, xs, [([h["avg_score"] for h in history], ACCENT, "avg score")],
                   y_fmt=lambda v: f"{v / 1000:.0f}k" if v >= 10_000 else f"{v / 1000:.1f}k" if v else "0")
        draw_chart(self.rate_chart, xs,
                   [([100 * h["rates"].get(t, 0) for h in history], color, t) for t, color in RATE_SERIES],
                   y_max=100, y_fmt=lambda v: f"{v:.0f}%", guide=95)


def draw_chart(c, xs, series, y_max=None, y_fmt=str, guide=None):
    """Simple line chart on a tk Canvas. series: [(ys, color, label)]."""
    c.delete("all")
    w, h = c.winfo_width(), c.winfo_height()
    if w < 50 or h < 50:
        return
    left, right, top, bottom = 46, 10, 20, 22
    if len(xs) < 2:
        c.create_text(w / 2, h / 2, text="Chart starts once a full window of games has finished",
                      font=FONT_SMALL, fill=MUTED)
        return
    x0, x1 = xs[0], xs[-1]
    if x1 == x0:
        x1 = x0 + 1
    top_val = y_max if y_max is not None else max(max(ys) for ys, _, _ in series) * 1.1 or 1
    pw, ph = w - left - right, h - top - bottom

    def px(x):
        return left + (x - x0) / (x1 - x0) * pw

    def py(y):
        return top + ph - y / top_val * ph

    for i in range(5):
        v = top_val * i / 4
        y = py(v)
        c.create_line(left, y, w - right, y, fill=GRID)
        c.create_text(left - 4, y, text=y_fmt(v), anchor="e", font=FONT_SMALL, fill=MUTED)
    for x, anchor in ((x0, "nw"), ((x0 + x1) / 2, "n"), (x1, "ne")):
        c.create_text(px(x), h - bottom + 4, text=fmt_count(int(x)), anchor=anchor,
                      font=FONT_SMALL, fill=MUTED)
    if guide is not None:
        c.create_line(left, py(guide), w - right, py(guide), fill=GOOD, dash=(2, 3))

    lx = left + 4
    for ys, color, label in series:
        pts = []
        for x, y in zip(xs, ys):
            pts += [px(x), py(min(y, top_val))]
        c.create_line(*pts, fill=color, width=2)
        c.create_rectangle(lx, 6, lx + 10, 14, fill=color, outline="")
        c.create_text(lx + 14, 10, text=label, anchor="w", font=FONT_SMALL, fill=TEXT)
        lx += 24 + 7 * len(label)


def main():
    parser = argparse.ArgumentParser(description="Live dashboard for 2048 agent training.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS,
                        help="the weights file being trained (stats are read from next to it)")
    args = parser.parse_args()
    root = tk.Tk()
    root.title("2048 Training Stats")
    root.configure(bg=BG)
    root.geometry("900x820")
    root.minsize(760, 700)
    ttk.Style().theme_use("default")
    dash = StatsDashboard(root, args.weights)
    dash.pack(fill="both", expand=True)
    on_top = tk.BooleanVar(value=False)
    tk.Checkbutton(dash.header, text="Keep on top", variable=on_top, bg=BG, fg=TEXT,
                   activebackground=BG, font=FONT_SMALL,
                   command=lambda: root.attributes("-topmost", on_top.get())).pack(side="right")
    root.mainloop()


if __name__ == "__main__":
    main()
