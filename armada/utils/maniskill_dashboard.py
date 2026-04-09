from __future__ import annotations

import os
import sys
import shutil
import threading
import time
import textwrap
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import cv2
import numpy as np


@dataclass
class ConsoleMessage:
    timestamp: float
    level: str
    text: str


class SimpleConsoleRolloutUI:
    """Minimal terminal UI for rollout status and message queue."""

    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.episode_idx = 0
        self.env_states: Dict[int, Dict[str, str]] = {}
        self.messages = deque(maxlen=max_messages)
        self._lock = threading.Lock()
        self._buffer = ""
        self._last_render = 0.0
        self._render_interval_s = 0.05
        self._terminal = sys.__stdout__

    def set_episode(self, episode_idx: int):
        with self._lock:
            self.episode_idx = int(episode_idx)

    def update_env(self, env_idx: int, *, step: int, state: str, decision: str, mode: str = "", round_idx: Optional[int] = None):
        with self._lock:
            payload = {
                "step": str(int(step)),
                "state": state,
                "decision": decision,
                "mode": mode,
            }
            if round_idx is not None:
                payload["round"] = str(int(round_idx))
            self.env_states[int(env_idx)] = payload
        self.render()

    def push_message(self, text: str, level: str = "info"):
        cleaned = str(text).strip()
        if not cleaned:
            return
        with self._lock:
            self.messages.append(ConsoleMessage(time.time(), level, cleaned))
        self.render()

    def write(self, text: str):
        if not text:
            return 0
        with self._lock:
            self._buffer += text
            lines = self._buffer.splitlines(keepends=True)
            self._buffer = ""
            for line in lines:
                if line.endswith("\n") or line.endswith("\r"):
                    cleaned = line.strip()
                    if cleaned:
                        self.messages.append(ConsoleMessage(time.time(), "info", cleaned))
                else:
                    self._buffer = line
        self.render()
        return len(text)

    def flush(self):
        with self._lock:
            if self._buffer.strip():
                self.messages.append(ConsoleMessage(time.time(), "info", self._buffer.strip()))
                self._buffer = ""
        self.render(force=True)

    def isatty(self):
        return True

    def fileno(self):
        return self._terminal.fileno()

    def _format_env_line(self, env_idx: int, state: Dict[str, str]) -> str:
        return (
            f"[{env_idx}] step={state.get('step', '-'):<4} "
            f"state={state.get('state', '-'):<20} "
            f"decision={state.get('decision', '-'):<10} "
            f"mode={state.get('mode', '-'):<8}"
        )

    def render(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_render < self._render_interval_s:
            return
        with self._lock:
            self._last_render = now
            width = shutil.get_terminal_size((120, 40)).columns
            hr = "─" * max(20, width - 2)
            lines = [
                f" ARMADA ManiSkill Console UI ".center(width, "─"),
                f" episode={self.episode_idx}  envs={len(self.env_states)}  messages={len(self.messages)} ",
                hr,
                " Environments ",
            ]
            if self.env_states:
                for env_idx in sorted(self.env_states.keys()):
                    lines.append("  " + self._format_env_line(env_idx, self.env_states[env_idx]))
            else:
                lines.append("  (no active environments)")
            lines.extend([
                hr,
                " Messages ",
            ])
            if self.messages:
                for msg in list(self.messages)[-self.max_messages :]:
                    stamp = time.strftime("%H:%M:%S", time.localtime(msg.timestamp))
                    prefix = {"error": "ERR", "warn": "WRN"}.get(msg.level, "INF")
                    lines.append(f"  {stamp} {prefix} {msg.text}")
            else:
                lines.append("  (no messages)")
            lines.append(hr)

            self._terminal.write("\033[2J\033[H")
            self._terminal.write("\n".join(lines) + "\n")
            self._terminal.flush()


class ManiSkillRolloutDashboard:
    """OpenCV-based dashboard for ManiSkill rollout monitoring.

    The dashboard is intentionally lightweight so it can run inside the same
    rollout process without additional dependencies.
    """

    def __init__(
        self,
        enabled: bool = True,
        window_name: str = "ARMADA ManiSkill Rollout",
        panel_width: int = 640,
        panel_height: int = 360,
        max_error_lines: int = 18,
    ):
        self.enabled = enabled
        self.window_name = window_name
        self.panel_width = panel_width
        self.panel_height = panel_height
        self.max_error_lines = max_error_lines
        self._window_ready = False
        self._display_available = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if self.enabled and not self._display_available:
            self.enabled = False

    def _ensure_window(self):
        if not self.enabled or self._window_ready:
            return
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            self._window_ready = True
        except Exception as exc:
            self.enabled = False
            self._window_ready = False
            _ = exc

    def _resize_rgb(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            return np.zeros((self.panel_height, self.panel_width, 3), dtype=np.uint8)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.resize(image, (self.panel_width, self.panel_height), interpolation=cv2.INTER_AREA)

    def _draw_text_block(
        self,
        canvas: np.ndarray,
        lines: Sequence[str],
        origin=(16, 28),
        color=(255, 255, 255),
        scale: float = 0.55,
        thickness: int = 1,
    ):
        x, y = origin
        for line in lines:
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
            y += int(22 * max(scale / 0.55, 1.0))

    def _truncate_error(self, error_text: Optional[str]) -> Sequence[str]:
        if not error_text:
            return ["No error reported."]
        wrapped = []
        for line in str(error_text).splitlines():
            wrapped.extend(textwrap.wrap(line, width=88) or [""])
        return wrapped[: self.max_error_lines]

    def _make_env_panel(self, env_idx: int, payload: Dict[str, np.ndarray], status: Dict[str, str]) -> np.ndarray:
        side = self._resize_rgb(payload.get("side_img"))
        wrist = self._resize_rgb(payload.get("wrist_img"))
        panel = np.vstack([side, wrist])

        header_bg = (0, 0, 0)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), header_bg, -1)
        title = f"Env {env_idx} | step={status.get('step', '-') } | mode={status.get('mode', '-') }"
        subtitle = f"state={status.get('state', '-') } | decision={status.get('decision', '-') }"
        cv2.putText(panel, title, (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, subtitle, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (230, 230, 230), 1, cv2.LINE_AA)

        error_reason = status.get("error_reason")
        if error_reason:
            cv2.rectangle(panel, (0, panel.shape[0] - 44), (panel.shape[1], panel.shape[0]), (0, 0, 120), -1)
            self._draw_text_block(panel, [f"Error: {error_reason}"], origin=(12, panel.shape[0] - 18), color=(255, 255, 255), scale=0.52)

        return panel

    def show(
        self,
        env_payloads: Dict[int, Dict[str, np.ndarray]],
        env_status: Dict[int, Dict[str, str]],
        banner: str = "",
        error_text: Optional[str] = None,
    ):
        if not self.enabled:
            return

        self._ensure_window()
        env_panels = []
        for env_idx in sorted(env_payloads.keys()):
            env_panels.append(self._make_env_panel(env_idx, env_payloads[env_idx], env_status.get(env_idx, {})))

        if not env_panels:
            return

        if len(env_panels) == 1:
            montage = env_panels[0]
        else:
            rows = []
            for i in range(0, len(env_panels), 2):
                row = env_panels[i]
                if i + 1 < len(env_panels):
                    right = env_panels[i + 1]
                    target_height = max(row.shape[0], right.shape[0])
                    if row.shape[0] != target_height:
                        row = cv2.copyMakeBorder(row, 0, target_height - row.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
                    if right.shape[0] != target_height:
                        right = cv2.copyMakeBorder(right, 0, target_height - right.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
                    row = np.hstack([row, right])
                rows.append(row)

            target_width = max(panel.shape[1] for panel in rows)
            padded_rows = []
            for row in rows:
                if row.shape[1] < target_width:
                    row = cv2.copyMakeBorder(row, 0, 0, 0, target_width - row.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
                padded_rows.append(row)
            montage = np.vstack(padded_rows)

        top_bar = np.zeros((72, montage.shape[1], 3), dtype=np.uint8)
        self._draw_text_block(
            top_bar,
            [banner or "ARMADA ManiSkill Rollout Dashboard", "Keys: C continue | T teleop | D discard | F finish | Q quit"],
            origin=(14, 26),
            color=(255, 255, 255),
            scale=0.6,
        )
        if error_text:
            self._draw_text_block(top_bar, ["Latest error:"] + self._truncate_error(error_text), origin=(14, 52), color=(0, 180, 255), scale=0.42)

        if top_bar.shape[1] != montage.shape[1]:
            top_bar = cv2.resize(top_bar, (montage.shape[1], top_bar.shape[0]))

        canvas = np.vstack([top_bar, montage])
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def wait_for_key(self, valid_keys: Sequence[str], timeout_ms: int = 0) -> Optional[str]:
        if not self.enabled:
            return None

        valid = {key.lower() for key in valid_keys}
        key_code = cv2.waitKey(timeout_ms if timeout_ms >= 0 else 1)
        if key_code == -1:
            return None

        try:
            key = chr(key_code & 0xFF).lower()
        except ValueError:
            return None

        return key if key in valid else None

    def show_error(self, error_text: str, banner: str = "Runtime error"):
        if not self.enabled:
            return

        self._ensure_window()
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 72), (0, 0, 120), -1)
        self._draw_text_block(canvas, [banner, "Press Q to quit, R to retry, D to discard current episode"], origin=(16, 28), color=(255, 255, 255), scale=0.7)
        lines = self._truncate_error(error_text)
        self._draw_text_block(canvas, lines, origin=(16, 120), color=(255, 220, 180), scale=0.5)
        cv2.imshow(self.window_name, canvas)
        cv2.waitKey(1)

    def close(self):
        if self.enabled and self._window_ready:
            cv2.destroyWindow(self.window_name)
            self._window_ready = False