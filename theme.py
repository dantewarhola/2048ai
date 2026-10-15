"""The app's look: colors, fonts and the widgets every page is built from.

Shared by app.py and stats_viewer.py so both windows match. Tk has no rounded
corners, shadows or custom sliders of its own, so cards, buttons, sliders,
segmented controls and status pills are drawn on canvases here.
"""

import math
import tkinter as tk
from tkinter import font as tkfont

# ---------------------------------------------------------------- colors

BG = "#f4f2ee"            # page background
PANEL = "#ffffff"         # cards
SUBTLE = "#f8f7f4"        # insets, hover on white
BORDER = "#e6e2db"        # hairlines around cards and inputs
BORDER_STRONG = "#d6d0c6"
SHADOW = "#e9e6e0"        # the 2px "lift" under cards
GRID = BORDER             # chart gridlines

TEXT = "#1f1c18"          # headings and values
TEXT_2 = "#5b554d"        # body text
MUTED = "#8c857b"         # captions and hints

INK = "#1f1c18"           # primary buttons
INK_HOVER = "#3a352f"
BRAND = "#f2b01e"         # 2048 gold
BRAND_HOVER = "#f5bf47"
BRAND_DEEP = "#d99a0b"
BRAND_SOFT = "#fdf3d8"
ACCENT = BRAND_DEEP       # charts and progress bars

GOOD = "#1e9b52"
GOOD_SOFT = "#e6f5ec"
WARN = "#b97409"
WARN_SOFT = "#fcf1d9"
BAD = "#d6402f"
BAD_HOVER = "#e0584a"
BAD_SOFT = "#fcebe8"
INFO = "#2f6fd6"
INFO_SOFT = "#e8f0fc"

DISABLED_BG = "#eeebe6"
DISABLED_TEXT = "#aea89f"

SIDEBAR = "#1c1a17"
SIDEBAR_HOVER = "#262320"
SIDEBAR_ACTIVE = "#2f2b27"
SIDEBAR_TEXT = "#f4f1ec"
SIDEBAR_MUTED = "#958d82"

CONSOLE_BG = "#1c1a17"
CONSOLE_TEXT = "#d8d2c8"
CONSOLE_MUTED = "#8a8278"
CONSOLE_BAR = "#151311"

# Status pill tones: (text/dot, background).
TONES = {
    "neutral": (TEXT_2, "#efece7"),
    "good": (GOOD, GOOD_SOFT),
    "warn": (WARN, WARN_SOFT),
    "bad": (BAD, BAD_SOFT),
    "info": (INFO, INFO_SOFT),
    "brand": (BRAND_DEEP, BRAND_SOFT),
}

# ---------------------------------------------------------------- type

FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI Semibold", 10)
FONT_SMALL_BOLD = ("Segoe UI Semibold", 9)
FONT_CAPS = ("Segoe UI Semibold", 8)      # small uppercase labels
FONT_H2 = ("Segoe UI Semibold", 12)
FONT_H1 = ("Segoe UI Semibold", 20)
FONT_STAT = ("Segoe UI Semibold", 20)
FONT_BIG = ("Segoe UI Semibold", 22)
FONT_HERO = ("Segoe UI Black", 30)
FONT_MONO = ("Consolas", 9)

# ---------------------------------------------------------------- scale

UI_SCALE = 1.0  # screen DPI / 96, set by set_scale() when the window opens


def set_scale(scale):
    global UI_SCALE
    UI_SCALE = scale


def px(n):
    """Scale a size in pixels for high-DPI screens."""
    return int(round(n * UI_SCALE))


def round_rect(canvas, x0, y0, x1, y1, r, **kw):
    """A rounded rectangle (a smoothed polygon, the usual Tk trick)."""
    r = max(0, min(r, (x1 - x0) / 2, (y1 - y0) / 2))
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(pts, smooth=True, **kw)


def bg_of(widget):
    try:
        return widget.cget("bg")
    except tk.TclError:
        return BG


# ---------------------------------------------------------------- widgets

class WrapLabel(tk.Label):
    """A label whose text wraps to however wide the layout makes it."""

    def __init__(self, parent, **kw):
        # Start narrow so the label never asks for more room than a sidebar
        # column has; it then widens to whatever space the layout gives it.
        kw.setdefault("wraplength", px(300))
        super().__init__(parent, **kw)
        self.bind("<Configure>", lambda e: self.config(wraplength=max(50, e.width - 4)))


class Card(tk.Frame):
    """A white panel with rounded corners, a hairline border and a slight lift.

    Put widgets inside it like any frame. It lives inside an outer "shell"
    frame whose canvas draws the rounded shape, so pack/grid/place and
    destroy are forwarded to that shell.
    """

    def __init__(self, parent, title=None, subtitle=None, padx=18, pady=16, bg=PANEL,
                 border=BORDER, radius=12, lift=True):
        self.fill, self.border, self.radius = bg, border, px(radius)
        self.lift = px(2) if lift else 0
        outer = bg_of(parent)
        self.shell = tk.Frame(parent, bg=outer)
        self.canvas = tk.Canvas(self.shell, bg=outer, highlightthickness=0, bd=0)
        self.canvas.place(x=0, y=0, relwidth=1, relheight=1)
        # Keep the content clear of the rounded corners.
        inset = math.ceil(self.radius * 0.3) + 1
        super().__init__(self.shell, bg=bg, padx=max(0, px(padx) - inset), pady=max(0, px(pady) - inset))
        tk.Frame.pack_configure(self, fill="both", expand=True, padx=inset,
                                pady=(inset, inset + self.lift))
        self.shell.bind("<Configure>", self._draw)
        if title:
            tk.Label(self, text=title, font=FONT_H2, bg=bg, fg=TEXT, anchor="w").pack(fill="x")
        if subtitle:
            WrapLabel(self, text=subtitle, font=FONT_SMALL, bg=bg, fg=MUTED, anchor="w",
                      justify="left").pack(fill="x", pady=(2, 8))

    def _draw(self, _event=None):
        c = self.canvas
        c.delete("all")
        w, h = self.shell.winfo_width(), self.shell.winfo_height()
        if w < 4 or h < 4:
            return
        if self.lift:
            round_rect(c, 0, self.lift, w - 1, h - 1, self.radius, fill=SHADOW, outline="")
        round_rect(c, 0, 0, w - 1, h - 1 - self.lift, self.radius, fill=self.fill,
                   outline=self.border or self.fill)

    # Geometry and lifetime belong to the shell.
    def pack_configure(self, *args, **kw):
        self.shell.pack_configure(*args, **kw)
    pack = pack_configure

    def grid_configure(self, *args, **kw):
        self.shell.grid_configure(*args, **kw)
    grid = grid_configure

    def place_configure(self, *args, **kw):
        self.shell.place_configure(*args, **kw)
    place = place_configure

    def pack_forget(self):
        self.shell.pack_forget()
    forget = pack_forget

    def grid_forget(self):
        self.shell.grid_forget()

    def grid_remove(self):
        self.shell.grid_remove()

    def destroy(self):
        tk.Frame.destroy(self)
        if not getattr(self, "_shell_gone", False):  # the shell's own destroy comes back here
            self._shell_gone = True
            self.shell.destroy()


class FlatButton(tk.Canvas):
    """A rounded button with hover, press and disabled states.

    kind: "primary" (dark), "gold" (brand), "secondary" (outlined), "danger",
    or "ghost" (text only until hovered).
    """

    STYLES = {  # background, text, hover background, border
        "primary": (INK, "#ffffff", INK_HOVER, None),
        "gold": (BRAND, INK, BRAND_HOVER, None),
        "secondary": (PANEL, TEXT, SUBTLE, BORDER_STRONG),
        "danger": (BAD, "#ffffff", BAD_HOVER, None),
        "ghost": (None, TEXT_2, "#efece7", None),
    }

    def __init__(self, parent, text, command=None, kind="primary", small=False, anchor="center", **kw):
        self.outer = bg_of(parent)
        self.kind = kind
        self.text = text
        self.command = command
        self.anchor = anchor
        self.small = small
        self.font = tkfont.Font(font=FONT_SMALL_BOLD if small else FONT_BOLD)
        self.padx = px(12) if small else px(18)
        self.radius = px(7) if small else px(9)
        self.enabled = True
        self.hover = False
        self.pressed = False
        super().__init__(parent, height=px(30) if small else px(38), bg=self.outer,
                         highlightthickness=0, bd=0, cursor="hand2", takefocus=1, **kw)
        self._fit()
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: self._set(hover=True))
        self.bind("<Leave>", lambda e: self._set(hover=False, pressed=False))
        self.bind("<ButtonPress-1>", lambda e: self._set(pressed=True))
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Return>", self._click)
        self.bind("<space>", self._click)

    def _fit(self):
        tk.Canvas.configure(self, width=self.font.measure(self.text) + 2 * self.padx)

    def configure(self, cnf=None, **kw):
        if "text" in kw:
            self.text = kw.pop("text")
            self._fit()
            self._draw()
        if cnf or kw:
            return tk.Canvas.configure(self, cnf, **kw)
    config = configure

    def set_kind(self, kind):
        self.kind = kind
        self._draw()

    def _set(self, **state):
        for k, v in state.items():
            setattr(self, k, v)
        self._draw()

    def _release(self, event):
        was_pressed = self.pressed
        self._set(pressed=False)
        inside = 0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height()
        if was_pressed and inside:
            self._click()

    def _click(self, _event=None):
        if self.enabled and self.command:
            self.command()

    def set_enabled(self, enabled):
        self.enabled = enabled
        tk.Canvas.configure(self, cursor="hand2" if enabled else "arrow")
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4:
            w, h = int(self.cget("width")), int(self.cget("height"))
        bg, fg, hover, border = self.STYLES[self.kind]
        if not self.enabled:
            bg, fg, border = DISABLED_BG, DISABLED_TEXT, None
        elif self.hover:
            bg = hover
        if bg is None:
            bg = self.outer
        top = 1 if self.pressed and self.enabled else 0  # a 1px "press" nudge
        round_rect(self, 0, top, w - 1, h - 1, self.radius, fill=bg, outline=border or bg)
        if self.anchor == "w":
            self.create_text(self.padx, h / 2 + top, text=self.text, anchor="w", fill=fg, font=self.font)
        else:
            self.create_text(w / 2, h / 2 + top, text=self.text, fill=fg, font=self.font)


class Slider(tk.Canvas):
    """A slider with a rounded track, a gold fill and a round knob."""

    def __init__(self, parent, var, frm, to, integer=False, command=None):
        self.outer = bg_of(parent)
        super().__init__(parent, height=px(24), bg=self.outer, highlightthickness=0, bd=0,
                         cursor="hand2")
        self.var, self.frm, self.to, self.integer, self.command = var, frm, to, integer, command
        self.hover = False
        self.dragging = False
        self.knob = px(8)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Enter>", lambda e: self._hover(True))
        self.bind("<Leave>", lambda e: self._hover(False))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._move)
        self.bind("<ButtonRelease-1>", self._release)
        var.trace_add("write", lambda *a: self._draw())

    def _hover(self, inside):
        self.hover = inside
        self._draw()

    def _value_at(self, x):
        span = max(1, self.winfo_width() - 2 * self.knob)
        frac = min(1.0, max(0.0, (x - self.knob) / span))
        value = self.frm + frac * (self.to - self.frm)
        return round(value) if self.integer else value

    def _press(self, event):
        self.dragging = True
        self._move(event)

    def _move(self, event):
        self.var.set(self._value_at(event.x))
        if self.command:
            self.command()

    def _release(self, _event):
        self.dragging = False
        self._draw()

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 10:
            return
        try:
            v = float(self.var.get())
        except (tk.TclError, ValueError):
            v = self.frm
        frac = (min(self.to, max(self.frm, v)) - self.frm) / ((self.to - self.frm) or 1)
        k, cy = self.knob, h / 2
        x = k + frac * (w - 2 * k)
        thick = px(4)
        self.create_line(k, cy, w - k, cy, fill="#e8e4dd", width=thick, capstyle="round")
        self.create_line(k, cy, x, cy, fill=BRAND, width=thick, capstyle="round")
        r = k + (1 if self.hover or self.dragging else 0)
        self.create_oval(x - r, cy - r + 1, x + r, cy + r + 1, fill=SHADOW, outline="")
        self.create_oval(x - r, cy - r, x + r, cy + r, fill=PANEL,
                         outline=BRAND_DEEP if self.dragging else BORDER_STRONG, width=1)


class SliderField:
    """A Slider with its current value shown next to it."""

    def __init__(self, parent, var, frm, to, fmt="{:.0f}", integer=False):
        self.var, self.fmt, self.integer = var, fmt, integer
        Slider(parent, var, frm, to, integer=integer).pack(side="left", fill="x", expand=True)
        self.label = tk.Label(parent, font=FONT_BOLD, bg=bg_of(parent), fg=TEXT, width=6, anchor="e")
        self.label.pack(side="left", padx=(10, 0))
        var.trace_add("write", lambda *a: self._show())
        self._show()

    def _show(self):
        try:
            value = float(self.var.get())
        except (tk.TclError, ValueError):
            return
        if self.integer and value != round(value):
            self.var.set(round(value))  # snap to whole steps
            return
        self.label.config(text=self.fmt.format(value))


class Segmented(tk.Canvas):
    """A segmented control: a row of options, one selected (like iOS / Fluent)."""

    def __init__(self, parent, var, options, command=None):
        """options: [(value, label)]."""
        self.outer = bg_of(parent)
        super().__init__(parent, height=px(34), bg=self.outer, highlightthickness=0, bd=0, cursor="hand2")
        self.var, self.options, self.command = var, options, command
        self.hover = None
        self.font = tkfont.Font(font=FONT_SMALL)
        self.font_sel = tkfont.Font(font=FONT_SMALL_BOLD)
        self.bind("<Configure>", lambda e: self._draw())
        self.bind("<Motion>", lambda e: self._hover(self._index_at(e.x)))
        self.bind("<Leave>", lambda e: self._hover(None))
        self.bind("<Button-1>", self._click)
        var.trace_add("write", lambda *a: self._draw())

    def _index_at(self, x):
        w = self.winfo_width()
        return min(len(self.options) - 1, max(0, int(x / (w / len(self.options)))))

    def _hover(self, i):
        if i != self.hover:
            self.hover = i
            self._draw()

    def _click(self, event):
        self.var.set(self.options[self._index_at(event.x)][0])
        if self.command:
            self.command()

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 20:
            return
        round_rect(self, 0, 0, w - 1, h - 1, px(9), fill="#efece7", outline="#efece7")
        seg = w / len(self.options)
        try:
            current = self.var.get()
        except tk.TclError:
            current = None
        pad = px(3)
        for i, (value, label) in enumerate(self.options):
            x0 = i * seg
            selected = value == current or (isinstance(current, float) and value == round(current))
            if selected:
                round_rect(self, x0 + pad, pad + 1, x0 + seg - pad, h - pad, px(7), fill=SHADOW, outline="")
                round_rect(self, x0 + pad, pad, x0 + seg - pad, h - pad - 1, px(7), fill=PANEL,
                           outline=BORDER)
            elif i == self.hover:
                round_rect(self, x0 + pad, pad, x0 + seg - pad, h - pad - 1, px(7), fill="#e5e1da",
                           outline="#e5e1da")
            self.create_text(x0 + seg / 2, h / 2, text=label, fill=TEXT if selected else TEXT_2,
                             font=self.font_sel if selected else self.font)


class StatusPill(tk.Canvas):
    """A rounded status badge with a colored dot that can pulse while active."""

    def __init__(self, parent, text="", tone="neutral"):
        self.outer = bg_of(parent)
        super().__init__(parent, height=px(26), bg=self.outer, highlightthickness=0, bd=0)
        self.font = tkfont.Font(font=FONT_SMALL_BOLD)
        self.text, self.tone = text, tone
        self.pulsing = False
        self.phase = 0
        self.set(text, tone)

    def set(self, text, tone="neutral", pulse=False):
        self.text, self.tone = text, tone
        tk.Canvas.configure(self, width=self.font.measure(text) + px(34))
        if pulse and not self.pulsing:
            self.pulsing = True
            self._pulse()
        self.pulsing = pulse
        self._draw()

    def _pulse(self):
        if not self.pulsing or not self.winfo_exists():
            self.phase = 0
            self._draw()
            return
        self.phase = (self.phase + 1) % 20
        self._draw()
        self.after(70, self._pulse)

    def _draw(self):
        self.delete("all")
        w, h = int(self.cget("width")), int(self.cget("height"))
        fg, bg = TONES.get(self.tone, TONES["neutral"])
        round_rect(self, 0, 0, w - 1, h - 1, h / 2, fill=bg, outline=bg)
        cx, cy, r = px(13), h / 2, px(4)
        if self.pulsing:  # a soft ring that grows and fades around the dot
            t = self.phase / 20
            ring = r + px(5) * t
            self.create_oval(cx - ring, cy - ring, cx + ring, cy + ring, outline=_mix(fg, bg, t), width=2)
        self.create_oval(cx - r, cy - r, cx + r, cy + r, fill=fg, outline="")
        self.create_text(px(23), cy, text=self.text, anchor="w", fill=fg, font=self.font)


def _mix(a, b, t):
    """Blend color a toward b by t (0..1)."""
    a = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


class ProgressBar(tk.Canvas):
    """A thin rounded progress bar; set it with config(value=...) like ttk's."""

    def __init__(self, parent, maximum=100, thickness=8, color=BRAND, **_ignored):
        self.outer = bg_of(parent)
        super().__init__(parent, height=px(thickness), bg=self.outer, highlightthickness=0, bd=0)
        self.maximum, self.value, self.color = maximum, 0, color
        self.bind("<Configure>", lambda e: self._draw())

    def configure(self, cnf=None, **kw):
        if "value" in kw:
            self.value = kw.pop("value")
            self._draw()
        if "maximum" in kw:
            self.maximum = kw.pop("maximum")
            self._draw()
        if cnf or kw:
            return tk.Canvas.configure(self, cnf, **kw)
    config = configure

    def _draw(self):
        self.delete("all")
        w, h = self.winfo_width(), self.winfo_height()
        if w < 4:
            return
        round_rect(self, 0, 0, w - 1, h - 1, h / 2, fill="#ebe7e0", outline="#ebe7e0")
        frac = min(1.0, max(0.0, self.value / (self.maximum or 1)))
        if frac > 0:
            round_rect(self, 0, 0, max(h, (w - 1) * frac), h - 1, h / 2, fill=self.color, outline=self.color)


class ScrollColumn(tk.Frame):
    """A fixed-width column whose content (self.inner) scrolls with the mouse
    wheel when it is taller than the window, with no visible scrollbar."""

    def __init__(self, parent, width):
        bg = bg_of(parent)
        super().__init__(parent, bg=bg)
        # A slim thumb at the right edge, shown only when there is more to scroll to.
        self.bar = tk.Canvas(self, width=px(6), bg=bg, highlightthickness=0, bd=0)
        self.bar.pack(side="right", fill="y", padx=(px(4), 0))
        self.bar.bind("<B1-Motion>", self._drag)
        self.bar.bind("<ButtonPress-1>", self._drag)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0, bd=0, yscrollincrement=px(24),
                                width=width, yscrollcommand=self._show_bar)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self.window = self.canvas.create_window(0, 0, window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._content_changed)
        self.canvas.bind("<Configure>", self._size_changed)
        self.bind_all("<MouseWheel>", self._wheel, add="+")

    def _content_changed(self, event):
        self.canvas.configure(scrollregion=(0, 0, event.width, event.height))
        self._clamp()

    def _size_changed(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)
        self._clamp()

    def _clamp(self):
        if self.inner.winfo_reqheight() <= self.canvas.winfo_height():
            self.canvas.yview_moveto(0)
        self._show_bar(*self.canvas.yview())

    def _show_bar(self, first, last):
        first, last = float(first), float(last)
        self.bar.delete("all")
        h = self.bar.winfo_height()
        if last - first >= 0.999 or h < 10:
            return
        w = int(self.bar.cget("width"))
        round_rect(self.bar, 0, first * h, w, max(first * h + w, last * h), w / 2,
                   fill=BORDER_STRONG, outline=BORDER_STRONG)

    def _drag(self, event):
        first, last = self.canvas.yview()
        self.canvas.yview_moveto(event.y / max(1, self.bar.winfo_height()) - (last - first) / 2)

    def _wheel(self, event):
        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None and widget is not self:
            widget = widget.master
        if widget is self and self.inner.winfo_reqheight() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-event.delta / 120) or (-1 if event.delta > 0 else 1), "units")


class StatGrid(tk.Frame):
    """Labeled numbers side by side, wrapping onto more rows when there isn't room."""

    def __init__(self, parent, names, cell_width=110, value_font=("Segoe UI Semibold", 16)):
        bg = bg_of(parent)
        super().__init__(parent, bg=bg)
        self.cell_width = cell_width
        self.cells, self.values, self.cols = [], {}, None
        for name in names:
            cell = tk.Frame(self, bg=bg)
            tk.Label(cell, text=name, font=FONT_SMALL, bg=bg, fg=MUTED, anchor="w").pack(fill="x")
            value = tk.Label(cell, text="-", font=value_font, bg=bg, fg=TEXT, anchor="w")
            value.pack(fill="x")
            self.cells.append(cell)
            self.values[name] = value
        self._layout(2)  # start narrow, so it never asks for more width than it gets
        self.bind("<Configure>", lambda e: self._layout(max(1, e.width // px(self.cell_width))))

    def _layout(self, cols):
        cols = min(cols, len(self.cells))
        if cols == self.cols:
            return
        self.cols = cols
        for i, cell in enumerate(self.cells):
            cell.grid(row=i // cols, column=i % cols, sticky="ew", pady=(0 if i < cols else px(8), 0))
        for c in range(len(self.cells)):
            self.columnconfigure(c, weight=1 if c < cols else 0, uniform="stat" if c < cols else "")

    def set(self, name, text):
        self.values[name].config(text=text)


def field(parent, label, hint=None):
    """A labeled row inside a card; returns the frame to put the control in."""
    bg = bg_of(parent)
    row = tk.Frame(parent, bg=bg)
    row.pack(fill="x", pady=(12, 0))
    tk.Label(row, text=label, font=FONT_BOLD, bg=bg, fg=TEXT, anchor="w").pack(fill="x")
    holder = tk.Frame(row, bg=bg)
    holder.pack(fill="x", pady=(5, 0))
    if hint:
        WrapLabel(row, text=hint, font=FONT_SMALL, bg=bg, fg=MUTED, anchor="w", justify="left"
                  ).pack(fill="x", pady=(4, 0))
    return holder


def divider(parent, pady=(12, 12)):
    tk.Frame(parent, bg=BORDER, height=1).pack(fill="x", pady=pady)
