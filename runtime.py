"""Where the app's code and data live, and how it starts its helper scripts.

The same code runs two ways:

- From the repo (python app.py): data stays in the project folder, and
  helper scripts run as `python -u script.py ...`.
- As a packaged "2048 AI.exe" (see build.py): the code is unpacked to a
  temporary folder, so models and settings live in %LOCALAPPDATA%\\2048 AI
  instead, and helpers run as `"2048 AI.exe" --run script.py ...`, which
  app.main() hands to run_script().
"""

import importlib
import os
import sys

FROZEN = getattr(sys, "frozen", False)
CODE_DIR = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))

if FROZEN:
    DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.expanduser("~"), "2048 AI")
    os.makedirs(DATA_DIR, exist_ok=True)
else:
    DATA_DIR = CODE_DIR

# Scripts the app runs as separate processes, and the modules that hold them.
SCRIPTS = {"agent2048.py": "agent2048", "ai_player.py": "ai_player", "read_board.py": "read_board"}


def python_exe():
    """python.exe even when the app runs under pythonw.exe (children need stdout)."""
    exe = sys.executable
    if os.path.basename(exe).lower() == "pythonw.exe":
        console = os.path.join(os.path.dirname(exe), "python.exe")
        if os.path.exists(console):
            return console
    return exe


def script_command(script, *args):
    """The command line that runs one of the project's scripts as its own process."""
    if FROZEN:
        return [sys.executable, "--run", script, *args]
    return [python_exe(), "-u", os.path.join(CODE_DIR, script), *args]


def _text_stream(fd, stream):
    """Line-buffered UTF-8 output, so the app sees each line as it is printed.

    A packaged exe ignores PYTHONUNBUFFERED / PYTHONIOENCODING, and a windowed
    one may start with no sys.stdout at all even though the app gave it a pipe.
    """
    if stream is not None:
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        return stream
    try:
        return open(fd, "w", encoding="utf-8", errors="replace", buffering=1, closefd=False)
    except OSError:
        return open(os.devnull, "w")


def run_script(argv):
    """Run `script.py args...` in this process (the exe's --run mode)."""
    script, args = argv[0], argv[1:]
    if script not in SCRIPTS:
        raise SystemExit(f"Unknown script: {script}")
    sys.stdout = _text_stream(1, sys.stdout)
    sys.stderr = _text_stream(2, sys.stderr)
    sys.argv = [script, *args]
    importlib.import_module(SCRIPTS[script]).main()
