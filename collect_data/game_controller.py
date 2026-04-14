"""
game_controller.py – Keyboard-based recording control.

Replaces the joystick-based GameController with a KeyboardController that
listens for key presses in a background thread using stdlib tty/termios
(no extra dependencies).

Key bindings:
    r  – start recording  (was: Square button)
    s  – save & stop recording  (was: Triangle button)
    q  – quit
"""

import select
import sys
import termios
import threading
import tty

from franka_robot import RobotInputs


class KeyboardController:
    """
    Non-blocking keyboard listener.

    Keys are fire-once: `get_recording_controls()` returns True for a key
    exactly once per press, then clears the flag.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._start_pressed = False  # 'r'
        self._stop_pressed = False   # 's'
        self._quit_pressed = False   # 'q' or Ctrl-C

        self._thread = threading.Thread(
            target=self._monitor_keys, daemon=True, name="keyboard-monitor"
        )
        self._thread.start()

        print("[KeyboardController] Keyboard listener started.")
        print("  'r' → start recording")
        print("  's' → save & stop recording")
        print("  'q' → quit")

    def _monitor_keys(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            while True:
                # Wait up to 0.1 s for a keypress so the thread can be killed
                readable, _, _ = select.select([sys.stdin], [], [], 0.1)
                if readable:
                    ch = sys.stdin.read(1)
                    with self._lock:
                        if ch == "r":
                            self._start_pressed = True
                        elif ch == "s":
                            self._stop_pressed = True
                        elif ch in ("q", "\x03"):  # 'q' or Ctrl-C
                            self._quit_pressed = True
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_recording_controls(self):
        """
        Returns (start_recording, stop_recording, quit).
        Flags are cleared after reading (fire-once semantics).
        """
        with self._lock:
            start = self._start_pressed
            stop = self._stop_pressed
            quit_ = self._quit_pressed
            self._start_pressed = False
            self._stop_pressed = False
            self._quit_pressed = False
        return start, stop, quit_
