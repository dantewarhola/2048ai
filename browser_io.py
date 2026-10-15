"""Reading the 2048 board from the screen and pressing keys in the browser.

BoardWatcher runs read_board.py and keeps the latest board it printed;
press_key sends a key press to whichever window has focus (Windows only).
"""

import ctypes
import subprocess
import threading
import time

from read_board import SIZE
from runtime import script_command

KEYS = {"W": 0x57, "A": 0x41, "S": 0x53, "D": 0x44}  # virtual-key codes
KEYEVENTF_KEYUP = 0x0002


def press_key(vk):
    """Send a key press to whichever window has focus."""
    user32 = ctypes.windll.user32
    scan = user32.MapVirtualKeyW(vk, 0)
    user32.keybd_event(vk, scan, 0, 0)
    time.sleep(0.01)
    user32.keybd_event(vk, scan, KEYEVENTF_KEYUP, 0)


def parse_cell(text):
    if text == ".":
        return 0
    if text == "?":
        return None
    return int(text)


class BoardWatcher:
    """Runs read_board.py and keeps track of the latest board it printed."""

    def __init__(self):
        self.proc = subprocess.Popen(
            script_command("read_board.py"),
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        self.lock = threading.Lock()
        self.board = None      # latest board printed
        self.visible = False   # whether read_board currently sees a board
        self.updates = 0       # number of boards printed so far
        self.finished = False  # read_board.py exited
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        rows = []
        for line in self.proc.stdout:
            print(line, end="", flush=True)  # echo read_board's output
            line = line.strip()
            if line.startswith("No 2048 board visible"):
                with self.lock:
                    self.visible = False
            elif line.startswith("|"):
                rows.append([parse_cell(c.strip()) for c in line.strip("|").split("|")])
                if len(rows) == SIZE:
                    with self.lock:
                        self.board = rows
                        self.visible = True
                        self.updates += 1
                    rows = []
        with self.lock:
            self.finished = True

    def state(self):
        with self.lock:
            return self.board, self.visible, self.updates, self.finished

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
