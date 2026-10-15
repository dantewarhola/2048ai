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

from agent2048 import (DEFAULT_WEIGHTS, STAT_TILES, eval_path, format_duration, read_json,
                       reliable_tile, stats_path, totals_from_history, wilson_interval)

REFRESH_MS = 1000
STALE_SECONDS = 15  # no stats update for this long while "training" = trainer not running

from theme import (ACCENT, BAD, BG, FONT_CAPS, GOOD, GRID, MUTED, PANEL, TEXT, Card, ProgressBar,
                   Segmented, WrapLabel, px)

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


def totals_table(totals):
    """tile_table() rows for all of a network's training games, from its totals."""
    n = totals["games"]
    rows = []
    for t in STAT_TILES:
        k = min(n, totals["tiles"].get(str(t), 0))
        lo, hi = wilson_interval(k, n)
        rows.append({"tile": t, "count": k, "rate": k / n if n else 0.0, "lo": lo, "hi": hi})
    return rows


class Panel(Card):
    def __init__(self, parent, title):
        super().__init__(parent, padx=12, pady=8)
        self.head = tk.Frame(self, bg=PANEL)
        self.head.pack(fill="x", pady=(0, px(4)))
        tk.Label(self.head, text=title, font=FONT_BOLD, bg=PANEL, fg=TEXT, anchor="w").pack(side="left")


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

        tiles = Panel(mid, "Highest tile reached")
        tiles.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        # All of the model's training games, or only the most recent ones.
        self.scope = tk.StringVar(value="all")
        self.scope_switch = Segmented(tiles.head, self.scope, [("all", "All training"), ("recent", "Recent")],
                                      command=self.scope_changed)
        self.scope_switch.config(width=px(190), height=px(30))
        self.scope_switch.pack(side="right")
        self.reliable = WrapLabel(tiles, text="", font=FONT, bg=PANEL, fg=TEXT, anchor="w", justify="left")
        self.reliable.pack(fill="x", pady=(2, 0))
        self.coverage = WrapLabel(tiles, text="", font=FONT_SMALL, bg=PANEL, fg=MUTED, anchor="w",
                                  justify="left")
        self.coverage.pack(fill="x")
        self.tile_canvas = tk.Canvas(tiles, height=8 * px(26) + px(6), bg=PANEL, highlightthickness=0)
        self.tile_canvas.pack(fill="x", pady=(px(6), 0))
        self.tile_canvas.bind("<Configure>", lambda e: self.draw_tiles())

        nums = Panel(mid, "Numbers")
        nums.grid(row=0, column=1, sticky="nsew")
        self.num_labels = {}
        grid = tk.Frame(nums, bg=PANEL)
        grid.pack(fill="x")
        groups = [("All training", ["Games", "Avg score", "Best score", "Best tile", "Avg moves / game",
                                    "Time training"]),
                  ("This run", ["Games this run", "Run time", "Games / sec", "Moves / sec",
                                "Last checkpoint", "Learning rate · batch"])]
        i = 0
        for g, (heading, rows) in enumerate(groups):
            tk.Label(grid, text=heading.upper(), font=FONT_CAPS, bg=PANEL, fg=MUTED, anchor="w"
                     ).grid(row=i, column=0, columnspan=2, sticky="w", pady=(px(10) if g else 0, px(2)))
            i += 1
            for name in rows:
                tk.Label(grid, text=name.replace(" this run", ""), font=FONT_SMALL, bg=PANEL, fg=MUTED,
                         anchor="w").grid(row=i, column=0, sticky="w", pady=1)
                lab = tk.Label(grid, text="-", font=FONT_BOLD, bg=PANEL, fg=TEXT, anchor="e")
                lab.grid(row=i, column=1, sticky="e", padx=(px(10), 0), pady=1)
                self.num_labels[name] = lab
                i += 1
        grid.columnconfigure(1, weight=1)

        # ---- eval (packed at the bottom first so the charts never push it off-screen)
        ev = Panel(outer, "Latest evaluation (with search)")
        ev.pack(side="bottom", fill="x", pady=(8, 0))
        self.eval_label = WrapLabel(ev, text="", font=FONT_SMALL, bg=PANEL, fg=TEXT,
                                    anchor="w", justify="left")
        self.eval_label.pack(fill="x")

        # ---- charts
        charts = tk.Frame(outer, bg=BG)
        charts.pack(fill="both", expand=True)
        charts.columnconfigure(0, weight=1)
        charts.columnconfigure(1, weight=1)
        charts.rowconfigure(0, weight=1)
        p1 = Panel(charts, "Average score over training")
        p1.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        p2 = Panel(charts, "% of games reaching tile")
        p2.grid(row=0, column=1, sticky="nsew")
        self.score_chart = tk.Canvas(p1, bg=PANEL, highlightthickness=0, height=px(190))
        self.score_chart.pack(fill="both", expand=True)
        self.rate_chart = tk.Canvas(p2, bg=PANEL, highlightthickness=0, height=px(190))
        self.rate_chart.pack(fill="both", expand=True)
        self.score_chart.bind("<Configure>", lambda e: self.draw_charts())
        self.rate_chart.bind("<Configure>", lambda e: self.draw_charts())

        self.refresh()

    def set_weights(self, weights):
        """Show the stats of another weights file."""
        self.stats_file = stats_path(weights)
        self.eval_file = eval_path(weights)
        self.stats = None
        self.totals = None
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
                        # Stats from before totals were kept: estimate them from the history.
                        self.totals = data.get("totals") or totals_from_history(data)
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

        t = self.totals
        n = self.num_labels
        n["Games"].config(text=f"{total:,}")
        n["Avg score"].config(text=f"{t['score_sum'] / t['games']:,.0f}" if t["games"] else "-")
        n["Best score"].config(text=f"{s.get('best_score_ever', 0):,}")
        n["Best tile"].config(text=f"{s.get('best_tile_ever', 0):,}")
        n["Avg moves / game"].config(
            text=f"{t['moves_sum'] / t['moves_games']:,.0f}" if t["moves_games"] else "-")
        n["Time training"].config(text=format_duration(t["seconds"]) if t["seconds"] else "-")
        n["Games this run"].config(text=f"{run:,}")
        n["Run time"].config(text=format_duration(s.get("elapsed", 0)))
        n["Games / sec"].config(text=f"{rate:,.1f}")
        n["Moves / sec"].config(text=f"{s.get('moves_per_sec', 0):,.0f}")
        n["Last checkpoint"].config(text=fmt_ago(s.get("last_save")))
        n["Learning rate · batch"].config(text=f"{s.get('alpha', 0):g} · {s.get('batch', 0):,}")
        self.update_tile_text()
        self.update_eval()

    def tile_rows(self):
        """The tile table for the chosen scope: all training, or the recent window."""
        if self.scope.get() == "all":
            return totals_table(self.totals) if self.totals and self.totals["games"] else []
        return (self.stats or {}).get("tiles") or []

    def scope_changed(self):
        self.draw_tiles()
        if self.stats:
            self.update_tile_text()

    def update_tile_text(self):
        s, t = self.stats, self.totals
        table = self.tile_rows()
        if self.scope.get() == "all":
            avg = t["score_sum"] / t["games"] if t["games"] else 0
            lines = [f"Avg score {avg:,.0f} over {t['games']:,} games"]
            notes = []
            missing = s.get("games_trained", 0) - t["games"]
            if missing > 0:
                notes.append(f"The first {missing:,} games weren't recorded, so they aren't included.")
            if t["estimated"]:
                notes.append(f"{t['estimated']:,} games from before totals were kept are estimated "
                             "from the chart history.")
            if t["moves_games"] < t["games"]:
                notes.append("Moves and training time are counted from then on.")
            self.coverage.config(text=" ".join(notes))
        else:
            lines = [f"Avg score {s.get('avg_score', 0):,.0f} over the last {s.get('window', 0):,} games "
                     f"(best {s.get('max_score_recent', 0):,})"]
            self.coverage.config(text="")
        rel = reliable_tile(table) if table else None
        lines.append(f"Reliable tile: {rel:,}  (≥95% of games, 95% confidence)" if rel else
                     "Reliable tile: none yet  (none reached in ≥95% of games with 95% confidence)")
        self.reliable.config(text="\n".join(lines))
        if self.coverage.cget("text"):
            self.coverage.pack(fill="x", before=self.tile_canvas)
        else:
            self.coverage.pack_forget()

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
        table = self.tile_rows()
        w = max(c.winfo_width(), px(300))
        if not table:
            c.create_text(px(8), px(20), text="Waiting for finished games...", anchor="w", font=FONT_SMALL,
                          fill=MUTED)
            return
        # The interval column is dropped when the panel is narrow.
        show_ci = w >= px(380)
        bar_x0, bar_x1 = px(70), w - px(150 if show_ci else 60)
        row_h = px(26)
        for i, row in enumerate(table):
            y = px(4) + i * row_h
            fg, bg = TILE_STYLE.get(row["tile"], ("#ffffff", ACCENT))
            c.create_rectangle(0, y, px(58), y + px(21), fill=bg, outline="")
            c.create_text(px(29), y + px(11), text=f"{row['tile']:,}", fill=fg, font=FONT_BOLD)
            # bar = observed rate, whisker = 95% confidence interval
            c.create_rectangle(bar_x0, y + px(5), bar_x1, y + px(16), fill=GRID, outline="")
            span = bar_x1 - bar_x0
            c.create_rectangle(bar_x0, y + px(5), bar_x0 + span * row["rate"], y + px(16), fill=bg, outline="")
            lo_x, hi_x = bar_x0 + span * row["lo"], bar_x0 + span * row["hi"]
            c.create_line(lo_x, y + px(10), hi_x, y + px(10), fill=TEXT, width=1)
            c.create_line(lo_x, y + px(6), lo_x, y + px(15), fill=TEXT)
            c.create_line(hi_x, y + px(6), hi_x, y + px(15), fill=TEXT)
            c.create_text(bar_x1 + px(8), y + px(11), anchor="w", font=FONT_BOLD, fill=TEXT,
                          text=f"{100 * row['rate']:.1f}%")
            if show_ci:
                c.create_text(bar_x1 + px(62), y + px(11), anchor="w", font=FONT_SMALL, fill=MUTED,
                              text=f"{100 * row['lo']:.1f}–{100 * row['hi']:.1f}%")
        # 95% marker
        x95 = bar_x0 + (bar_x1 - bar_x0) * 0.95
        c.create_line(x95, 0, x95, px(4) + len(table) * row_h, fill=GOOD, dash=(2, 3))

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
    left, right, top, bottom = px(46), px(10), px(20), px(22)
    if len(xs) < 2:
        c.create_text(w / 2, h / 2, text="Chart starts once a full window of games has finished",
                      font=FONT_SMALL, fill=MUTED, width=w - px(20), justify="center")
        return
    x0, x1 = xs[0], xs[-1]
    if x1 == x0:
        x1 = x0 + 1
    top_val = y_max if y_max is not None else max(max(ys) for ys, _, _ in series) * 1.1 or 1
    pw, ph = w - left - right, h - top - bottom

    def sx(x):
        return left + (x - x0) / (x1 - x0) * pw

    def sy(y):
        return top + ph - y / top_val * ph

    for i in range(5):
        v = top_val * i / 4
        y = sy(v)
        c.create_line(left, y, w - right, y, fill=GRID)
        c.create_text(left - px(4), y, text=y_fmt(v), anchor="e", font=FONT_SMALL, fill=MUTED)
    for x, anchor in ((x0, "nw"), ((x0 + x1) / 2, "n"), (x1, "ne")):
        c.create_text(sx(x), h - bottom + px(4), text=fmt_count(int(x)), anchor=anchor,
                      font=FONT_SMALL, fill=MUTED)
    if guide is not None:
        c.create_line(left, sy(guide), w - right, sy(guide), fill=GOOD, dash=(2, 3))

    lx = left + px(4)
    for ys, color, label in series:
        pts = []
        for x, y in zip(xs, ys):
            pts += [sx(x), sy(min(y, top_val))]
        c.create_line(*pts, fill=color, width=2)
        c.create_rectangle(lx, px(6), lx + px(10), px(14), fill=color, outline="")
        text = c.create_text(lx + px(14), px(10), text=label, anchor="w", font=FONT_SMALL, fill=TEXT)
        lx = c.bbox(text)[2] + px(12)


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
