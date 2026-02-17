#!/usr/bin/env python3
"""
CUA Mission Control V2 — With Planning LLM
Professional task control interface with OpenRouter planning model integration.
User gives simple commands → Planning LLM breaks them into steps → Qwen3-VL executes each step.
"""
from __future__ import annotations

import sys
import time
import threading
import traceback
from typing import Any, Dict, List, Optional, Tuple

from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPixmap, QImage, QPainter, QKeyEvent, QMouseEvent, QWheelEvent, QShortcut, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel,
    QVBoxLayout, QHBoxLayout, QFrame, QSizePolicy, QSplitter,
    QLineEdit, QComboBox, QPushButton, QGroupBox, QFormLayout,
    QTextEdit,
)

from src.config import cfg
from src.sandbox import Sandbox
from src.llm_client import load_llm, ask_next_action
from src.vision import capture_screen, capture_screen_raw, draw_preview
from src.guards import validate_xy, should_stop_on_repeat
from src.actions import execute_action
from src.design_system import build_stylesheet
from src.panels import TopBar, CommandPanel, InspectorPanel, LogPanel
from src.planner import PlannerConfig, create_planner, generate_plan, parse_plan_step
from src.agent_runner_v2 import run_planned_command


# ═══════════════════════════════════════════
# Agent core (from gui_main.py)
# ═══════════════════════════════════════════

def trim_history(history: List[Dict[str, Any]], keep_last: int = 6) -> List[Dict[str, Any]]:
    return history[-keep_last:] if len(history) > keep_last else history


def _center_from_bbox(b: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, b)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _extract_xy(out: Dict[str, Any]) -> Tuple[float, float]:
    x = out.get("x", 0.5)
    y = out.get("y", 0.5)
    pos = out.get("position", None)
    if pos is not None and isinstance(pos, (list, tuple)):
        if len(pos) == 2 and all(isinstance(t, (int, float)) for t in pos):
            return float(pos[0]), float(pos[1])
        if len(pos) == 4 and all(isinstance(t, (int, float)) for t in pos):
            return _center_from_bbox(list(pos))
        if len(pos) == 2 and all(isinstance(t, (list, tuple)) and len(t) == 2 for t in pos):
            (x1, y1), (x2, y2) = pos
            return (float(x1) + float(x2)) / 2.0, (float(y1) + float(y2)) / 2.0
    if isinstance(x, (list, tuple)):
        if len(x) == 2: return float(x[0]), float(x[1])
        if len(x) == 4: return _center_from_bbox(list(x))
    if isinstance(y, (list, tuple)):
        if len(y) == 2: return float(y[0]), float(y[1])
        if len(y) == 4: return _center_from_bbox(list(y))
    return float(x), float(y)


# ═══════════════════════════════════════════
# PIL → QPixmap
# ═══════════════════════════════════════════

def pil_to_qpixmap(pil_img) -> QPixmap:
    rgb = pil_img.convert("RGB")
    w, h = rgb.size
    data = rgb.tobytes("raw", "RGB")
    bpl = 3 * w
    qimg = QImage(data, w, h, bpl, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimg)


# ═══════════════════════════════════════════
# VMView — Live VM Screen
# ═══════════════════════════════════════════

class VMView(QLabel):
    """Renders the VM screen with letterbox scaling and forwards mouse/keyboard input."""

    def __init__(self, sandbox: Sandbox, parent=None):
        super().__init__(parent)
        self.sandbox = sandbox
        self._pm: Optional[QPixmap] = None
        self._draw_rect: Optional[Tuple[int, int, int, int]] = None
        self.input_enabled: bool = True
        self._pressed_btn: Optional[int] = None
        self._last_move_ts: float = 0.0
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setObjectName("vmView")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_frame(self, pm: QPixmap) -> None:
        self._pm = pm
        self.update()

    def _pos_to_norm(self, x: int, y: int) -> Optional[Tuple[float, float]]:
        if not self._pm or not self._draw_rect:
            return None
        dx, dy, dw, dh = self._draw_rect
        if dw <= 0 or dh <= 0:
            return None
        if x < dx or y < dy or x >= dx + dw or y >= dy + dh:
            return None
        return float((x - dx) / dw), float((y - dy) / dh)

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.GlobalColor.black)
        if not self._pm:
            p.end()
            return
        scaled = self._pm.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        self._draw_rect = (x, y, scaled.width(), scaled.height())
        p.drawPixmap(x, y, scaled)
        p.end()

    def mousePressEvent(self, e: QMouseEvent):
        if not self.input_enabled: return
        self.setFocus()
        mapped = self._pos_to_norm(int(e.position().x()), int(e.position().y()))
        if not mapped: return
        nx, ny = mapped
        btn_map = {Qt.MouseButton.LeftButton: 1, Qt.MouseButton.RightButton: 3, Qt.MouseButton.MiddleButton: 2}
        btn = btn_map.get(e.button())
        if btn is None: return
        self._pressed_btn = btn
        self.sandbox.mouse_move_norm(nx, ny)
        self.sandbox.mouse_down(btn)

    def mouseMoveEvent(self, e: QMouseEvent):
        if not self.input_enabled: return
        mapped = self._pos_to_norm(int(e.position().x()), int(e.position().y()))
        if not mapped: return
        nx, ny = mapped
        now = time.monotonic()
        if (now - self._last_move_ts) < 0.03: return
        self._last_move_ts = now
        if self._pressed_btn is not None:
            self.sandbox.drag_to_norm(nx, ny, self._pressed_btn)
        else:
            self.sandbox.mouse_move_norm(nx, ny)

    def mouseReleaseEvent(self, e: QMouseEvent):
        if not self.input_enabled or self._pressed_btn is None: return
        self.sandbox.mouse_up(self._pressed_btn)
        self._pressed_btn = None

    def wheelEvent(self, e: QWheelEvent):
        if not self.input_enabled: return
        self.sandbox.scroll(int(e.angleDelta().y()))

    def keyPressEvent(self, e: QKeyEvent):
        if not self.input_enabled: return
        if e.key() == Qt.Key.Key_F11:
            try: self.window().toggle_fullscreen()
            except: pass
            return
        mods = e.modifiers()
        txt = e.text() or ""
        if (mods & Qt.KeyboardModifier.ControlModifier) and txt and txt.isprintable():
            self.sandbox.hotkey(["ctrl", txt.lower()]); return
        if (mods & Qt.KeyboardModifier.AltModifier) and e.key() == Qt.Key.Key_Tab:
            self.sandbox.hotkey(["alt", "tab"]); return
        if txt and txt.isprintable() and len(txt) == 1:
            self.sandbox.type_text(txt); return
        special = {
            Qt.Key.Key_Return: "enter", Qt.Key.Key_Enter: "enter",
            Qt.Key.Key_Tab: "tab", Qt.Key.Key_Escape: "esc",
            Qt.Key.Key_Backspace: "backspace", Qt.Key.Key_Delete: "delete",
            Qt.Key.Key_Up: "up", Qt.Key.Key_Down: "down",
            Qt.Key.Key_Left: "left", Qt.Key.Key_Right: "right",
            Qt.Key.Key_Home: "home", Qt.Key.Key_End: "end",
            Qt.Key.Key_PageUp: "pageup", Qt.Key.Key_PageDown: "pagedown",
            Qt.Key.Key_Space: "space",
        }
        k = special.get(e.key())
        if k:
            self.sandbox.press_key(k)


# ═══════════════════════════════════════════
# Agent Signals
# ═══════════════════════════════════════════

class AgentSignals(QObject):
    log = pyqtSignal(str, str)          # msg, level
    busy = pyqtSignal(bool)
    finished = pyqtSignal(str)
    step_update = pyqtSignal(int, str, str)   # step_num, action, detail
    action_update = pyqtSignal(dict)
    latency_update = pyqtSignal(float)
    plan_ready = pyqtSignal(list)       # plan steps list


# ═══════════════════════════════════════════
# API Settings Panel
# ═══════════════════════════════════════════

class APISettingsPanel(QGroupBox):
    """Panel for configuring the Planning LLM (OpenRouter/OpenAI API)."""

    settings_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("🧠 Planning LLM Settings", parent)
        self.setObjectName("apiSettingsPanel")
        self.setStyleSheet("""
            QGroupBox {
                color: #e3f2fd;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #1e3a5f;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 18px;
                background: #0f2744;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
            }
            QLabel {
                color: #90caf9;
                font-size: 12px;
            }
            QLineEdit, QComboBox {
                background: #0d1f36;
                color: #e3f2fd;
                border: 1px solid #1565c0;
                border-radius: 4px;
                padding: 5px 8px;
                font-size: 12px;
            }
            QLineEdit:focus, QComboBox:focus {
                border-color: #2196f3;
            }
            QPushButton {
                background: #1565c0;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 14px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #1976d2;
            }
            QPushButton:pressed {
                background: #0d47a1;
            }
            QPushButton#testBtn {
                background: #2e7d32;
            }
            QPushButton#testBtn:hover {
                background: #388e3c;
            }
        """)

        layout = QFormLayout(self)
        layout.setContentsMargins(12, 24, 12, 12)
        layout.setSpacing(8)

        # Provider dropdown
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["OpenRouter", "OpenAI", "Local (Qwen3-VL only)"])
        self.provider_combo.currentIndexChanged.connect(self._on_provider_change)
        layout.addRow("Provider:", self.provider_combo)

        # Model input
        self.model_input = QLineEdit()
        self.model_input.setText(cfg.PLANNER_MODEL)
        self.model_input.setPlaceholderText("e.g. meta-llama/llama-3.3-70b-instruct:free")
        layout.addRow("Model:", self.model_input)

        # API Key input
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("Enter your API key...")
        self.api_key_input.setText(cfg.PLANNER_API_KEY)
        layout.addRow("API Key:", self.api_key_input)

        # Buttons row
        btn_row = QHBoxLayout()
        self.test_btn = QPushButton("🔗 Test Connection")
        self.test_btn.setObjectName("testBtn")
        self.test_btn.clicked.connect(self._on_test)
        btn_row.addWidget(self.test_btn)

        self.save_btn = QPushButton("💾 Apply Settings")
        self.save_btn.clicked.connect(self._on_save)
        btn_row.addWidget(self.save_btn)
        layout.addRow(btn_row)

        # Status label
        self.status_label = QLabel("⚪ Not configured")
        self.status_label.setObjectName("plannerStatus")
        layout.addRow(self.status_label)

        self.setFixedWidth(320)

    def _on_provider_change(self, idx: int):
        is_local = idx == 2  # "Local (Qwen3-VL only)"
        self.model_input.setEnabled(not is_local)
        self.api_key_input.setEnabled(not is_local)
        self.test_btn.setEnabled(not is_local)
        if is_local:
            self.status_label.setText("🟢 Using local Qwen3-VL only (no planner)")

    def _on_save(self):
        self.settings_changed.emit()
        self.status_label.setText("🟢 Settings applied")

    def _on_test(self):
        """Test the API connection with a simple request."""
        self.status_label.setText("🟡 Testing connection...")
        self.test_btn.setEnabled(False)

        config = self.get_config()
        def test_thread():
            try:
                planner = create_planner(config)
                if planner is None:
                    QTimer.singleShot(0, lambda: self._set_test_result(False, "No API key provided"))
                    return
                # Simple test: ask for a plan
                steps = generate_plan(planner, "test connection")
                if steps:
                    QTimer.singleShot(0, lambda: self._set_test_result(True, f"Connected! Got {len(steps)} steps"))
                else:
                    QTimer.singleShot(0, lambda: self._set_test_result(False, "Empty response"))
            except Exception as e:
                err = str(e)[:80]
                QTimer.singleShot(0, lambda: self._set_test_result(False, err))

        threading.Thread(target=test_thread, daemon=True).start()

    def _set_test_result(self, success: bool, msg: str):
        self.test_btn.setEnabled(True)
        if success:
            self.status_label.setText(f"🟢 {msg}")
        else:
            self.status_label.setText(f"🔴 {msg}")

    def get_config(self) -> PlannerConfig:
        provider_map = {0: "openrouter", 1: "openai", 2: "local"}
        return PlannerConfig(
            provider=provider_map.get(self.provider_combo.currentIndex(), "openrouter"),
            api_key=self.api_key_input.text().strip(),
            model=self.model_input.text().strip(),
            max_tokens=cfg.PLANNER_MAX_TOKENS,
        )


# ═══════════════════════════════════════════
# Plan Display Widget
# ═══════════════════════════════════════════

class PlanDisplayWidget(QFrame):
    """Shows the generated plan steps before/during execution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("planDisplay")
        self.setStyleSheet("""
            QFrame#planDisplay {
                background: #0a1929;
                border: 1px solid #1e3a5f;
                border-radius: 6px;
            }
            QLabel#planTitle {
                color: #64b5f6;
                font-weight: bold;
                font-size: 13px;
            }
            QTextEdit {
                background: transparent;
                color: #e3f2fd;
                border: none;
                font-size: 12px;
                font-family: 'Fira Code', 'Consolas', monospace;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        self.title = QLabel("📋 Generated Plan")
        self.title.setObjectName("planTitle")
        layout.addWidget(self.title)

        self.plan_text = QTextEdit()
        self.plan_text.setReadOnly(True)
        self.plan_text.setMaximumHeight(160)
        layout.addWidget(self.plan_text)

        self._steps: List[str] = []
        self._current_step: int = -1

    def set_plan(self, steps: List[str]):
        """Display the planned steps."""
        self._steps = steps
        self._current_step = -1
        self._render()

    def set_current_step(self, idx: int):
        """Highlight the currently executing step."""
        self._current_step = idx
        self._render()

    def _render(self):
        if not self._steps:
            self.plan_text.setHtml("<i style='color:#546e7a'>No plan generated yet</i>")
            return

        lines = []
        for i, step in enumerate(self._steps):
            parsed = parse_plan_step(step)
            verb = parsed["verb"].upper()
            target = parsed["target"]

            if i < self._current_step:
                # Completed
                icon = "✅"
                color = "#4caf50"
            elif i == self._current_step:
                # Currently executing
                icon = "▶️"
                color = "#ffb74d"
            else:
                # Pending
                icon = "⬜"
                color = "#546e7a"

            lines.append(
                f"<span style='color:{color}'>{icon} "
                f"<b>{i+1}.</b> <span style='color:#64b5f6'>{verb}</span> {target}</span>"
            )

        self.plan_text.setHtml("<br>".join(lines))

    def clear(self):
        self._steps = []
        self._current_step = -1
        self.plan_text.clear()


# ═══════════════════════════════════════════
# Direct runner (no planner — backward compatible)
# ═══════════════════════════════════════════

def run_single_command(
    sandbox: Sandbox, llm, objective: str,
    signals: AgentSignals,
    stop_event: Optional[threading.Event] = None,
) -> str:
    """Original single-command runner for local-only mode."""
    history: List[Dict[str, Any]] = []
    step = 1
    click_count = 0
    type_count = 0

    while True:
        if stop_event and stop_event.is_set():
            return "STOPPED"

        signals.log.emit(f"═══ STEP {step} ═══", "info")
        time.sleep(getattr(cfg, "WAIT_BEFORE_SCREENSHOT_SEC", 0.2))
        img = capture_screen(sandbox, cfg.SCREENSHOT_PATH)

        out: Optional[Dict[str, Any]] = None

        for attempt in range(getattr(cfg, "MODEL_RETRY", 2) + 1):
            out = ask_next_action(llm, objective, cfg.SCREENSHOT_PATH, trim_history(history))
            action = (out.get("action") or "NOOP").upper()
            if action == "BITTI":
                return "DONE(BITTI)"
            if action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
                x, y = _extract_xy(out)
                ok, reason = validate_xy(x, y)
                if ok:
                    out["x"], out["y"] = x, y
                    break
                signals.log.emit(f"[WARN] Invalid coordinates ({reason}), retrying.", "warn")
                history.append({"action": "INVALID_COORDS", "raw": out})
                out = None
                continue
            break

        if out is None:
            return "ERROR(no valid action)"

        action = (out.get("action") or "").upper()
        detail = out.get("why_short", out.get("target", ""))
        signals.log.emit(f"[MODEL] {action}: {detail}", "model")
        signals.action_update.emit(out)
        signals.step_update.emit(step, action, str(detail))

        if action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            click_count += 1
        if action == "TYPE":
            type_count += 1

        stop, why = should_stop_on_repeat(history, out)
        if stop:
            signals.log.emit(f"[STOPPED] {why}", "warn")
            return "DONE(repeat-guard)"

        if action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            preview_path = cfg.PREVIEW_PATH_TEMPLATE.format(i=step)
            draw_preview(img, float(out["x"]), float(out["y"]), preview_path)

        execute_action(sandbox, out)
        history.append(out)
        step += 1
        if step > getattr(cfg, "MAX_STEPS", 30):
            return "DONE(max-steps)"


# ═══════════════════════════════════════════
# MAIN WINDOW V2
# ═══════════════════════════════════════════

class MissionControlWindowV2(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CUA Mission Control V2 — With Planning LLM")
        self.resize(1680, 980)
        self.setStyleSheet(build_stylesheet())

        # --- State ---
        self.sandbox: Optional[Sandbox] = None
        self.llm = None
        self.planner = None
        self.planner_config = PlannerConfig()
        self.stop_event: Optional[threading.Event] = None
        self.worker_thread: Optional[threading.Thread] = None
        self._step_count = 0
        self._click_count = 0
        self._type_count = 0
        self._run_start: float = 0
        self._current_plan: List[str] = []

        # --- Signals ---
        self.signals = AgentSignals()
        self.signals.log.connect(self._on_log)
        self.signals.busy.connect(self._on_busy)
        self.signals.finished.connect(self._on_finished)
        self.signals.step_update.connect(self._on_step)
        self.signals.action_update.connect(self._on_action)
        self.signals.latency_update.connect(self._on_latency)
        self.signals.plan_ready.connect(self._on_plan_ready)

        # --- Build UI ---
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Top bar
        self.top_bar = TopBar()
        root_layout.addWidget(self.top_bar)

        # Middle: left | center | right
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(10, 8, 10, 0)
        body_layout.setSpacing(8)

        # LEFT COLUMN: Command panel + Plan display
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.cmd_panel = CommandPanel()
        self.cmd_panel.run_requested.connect(self._on_run)
        self.cmd_panel.stop_requested.connect(self._on_stop)
        left_layout.addWidget(self.cmd_panel)

        self.plan_display = PlanDisplayWidget()
        left_layout.addWidget(self.plan_display)

        # CENTER: VM view
        center_frame = QFrame()
        center_frame.setObjectName("centerPanel")
        center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(4, 4, 4, 4)

        self.vm_view_placeholder = QLabel("Connecting to sandbox…")
        self.vm_view_placeholder.setObjectName("vmView")
        self.vm_view_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.vm_view_placeholder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        center_layout.addWidget(self.vm_view_placeholder)
        self.vm_view: Optional[VMView] = None

        # RIGHT COLUMN: Inspector + API Settings
        right_col = QWidget()
        right_layout = QVBoxLayout(right_col)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.inspector = InspectorPanel()
        right_layout.addWidget(self.inspector)

        self.api_settings = APISettingsPanel()
        self.api_settings.settings_changed.connect(self._on_settings_changed)
        right_layout.addWidget(self.api_settings)

        body_layout.addWidget(left_col)
        body_layout.addWidget(center_frame, stretch=1)
        body_layout.addWidget(right_col)

        root_layout.addWidget(body, stretch=1)

        # Bottom log panel
        self.log_panel = LogPanel()
        log_wrap = QWidget()
        log_layout = QHBoxLayout(log_wrap)
        log_layout.setContentsMargins(10, 4, 10, 8)
        log_layout.addWidget(self.log_panel)
        root_layout.addWidget(log_wrap)

        # --- Keyboard Shortcuts ---
        QShortcut(QKeySequence("Ctrl+Return"), self, self._shortcut_run)
        QShortcut(QKeySequence("Escape"), self, self._on_stop)
        QShortcut(QKeySequence("F11"), self, self.toggle_fullscreen)
        QShortcut(QKeySequence("Ctrl+L"), self, self.log_panel.clear)

        # --- Timer for VM screenshots ---
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._refresh_vm)
        self.refresh_timer.start(350)

        # --- Init sandbox + model in background ---
        self._center_frame = center_frame
        self._center_layout = center_layout
        self.log_panel.append("Starting up… (V2 with Planning LLM)", "info")
        threading.Thread(target=self._init_backend, daemon=True).start()

    def _init_backend(self) -> None:
        # Sandbox
        try:
            self.signals.log.emit("Starting Docker container…", "info")
            self.sandbox = Sandbox(cfg)
            self.sandbox.start()
            self.signals.log.emit("Docker sandbox connected ✓", "success")
            QTimer.singleShot(0, self._setup_vm_view)
            QTimer.singleShot(0, lambda: self.top_bar.set_docker_status(True))
            QTimer.singleShot(0, lambda: self.inspector.set_vm_info(
                cfg.SANDBOX_NAME, cfg.VNC_RESOLUTION,
                f"http://127.0.0.1:{cfg.API_PORT}"))
        except Exception as e:
            self.signals.log.emit(f"Docker ERROR: {e}", "error")
            QTimer.singleShot(0, lambda: self.top_bar.set_docker_status(False))

        # LLM
        try:
            QTimer.singleShot(0, lambda: self.top_bar.set_model_status("loading"))
            self.signals.log.emit("Loading model (Qwen3-VL)…", "info")
            self.llm = load_llm()
            self.signals.log.emit("Qwen3-VL model ready ✓", "success")
            QTimer.singleShot(0, lambda: self.top_bar.set_model_status("ready"))
        except Exception as e:
            self.signals.log.emit(f"Model ERROR: {e}", "error")
            QTimer.singleShot(0, lambda: self.top_bar.set_model_status("error"))

        QTimer.singleShot(0, lambda: self.inspector.set_config(cfg))

    def _setup_vm_view(self) -> None:
        if not self.sandbox:
            return
        self.vm_view = VMView(self.sandbox)
        self.vm_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._center_layout.replaceWidget(self.vm_view_placeholder, self.vm_view)
        self.vm_view_placeholder.deleteLater()
        self.vm_view_placeholder = None

    def _refresh_vm(self) -> None:
        if not self.sandbox or not self.vm_view:
            return
        try:
            img = capture_screen_raw(self.sandbox)
            pm = pil_to_qpixmap(img)
            self.vm_view.set_frame(pm)
        except Exception:
            pass

    # --- Settings ---
    def _on_settings_changed(self) -> None:
        self.planner_config = self.api_settings.get_config()
        if self.planner_config.provider == "local":
            self.planner = None
            self.log_panel.append("Planner disabled — using direct Qwen3-VL mode.", "info")
        else:
            try:
                self.planner = create_planner(self.planner_config)
                if self.planner:
                    self.log_panel.append(
                        f"Planner configured: {self.planner_config.provider} / {self.planner_config.model}", "success")
                else:
                    self.log_panel.append("Planner requires an API key.", "warn")
            except Exception as e:
                self.log_panel.append(f"Planner setup error: {e}", "error")

    # --- Shortcuts ---
    def _shortcut_run(self) -> None:
        text = self.cmd_panel.cmd_input.text().strip()
        if text:
            self._on_run(text)

    def toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.showMaximized()
        else:
            self.showFullScreen()

    # --- Agent execution ---
    def _on_run(self, objective: str) -> None:
        if not objective:
            self.log_panel.append("Command cannot be empty.", "warn")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            self.log_panel.append("A command is already running.", "warn")
            return
        if not self.llm:
            self.log_panel.append("Qwen3-VL model not loaded yet!", "error")
            return
        if not self.sandbox:
            self.log_panel.append("No sandbox connection!", "error")
            return

        # Optional translation
        translated = objective
        try:
            from transformers import MarianMTModel, MarianTokenizer
            _tn = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-tc-big-tr-en")
            _tm = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-tc-big-tr-en")
            out = _tm.generate(**_tn(objective, return_tensors="pt", padding=True))
            translated = _tn.decode(out[0], skip_special_tokens=True)
            if translated != objective:
                self.log_panel.append(f"Translation: {objective} → {translated}", "info")
        except Exception:
            pass

        self._step_count = 0
        self._click_count = 0
        self._type_count = 0
        self._run_start = time.time()
        self.cmd_panel.clear_steps()
        self.plan_display.clear()
        self.stop_event = threading.Event()
        self.signals.busy.emit(True)

        # Decide: planned or direct mode
        use_planner = self.planner is not None and self.planner_config.provider != "local"

        if use_planner:
            self.log_panel.append(f"🧠 Sending to planner: \"{translated}\"", "info")
            self._run_with_planner(translated)
        else:
            self.log_panel.append(f"▶ Direct mode: \"{translated}\"", "info")
            self._run_direct(translated)

    def _run_with_planner(self, objective: str) -> None:
        """Phase 1: Generate plan, Phase 2: Execute plan steps."""
        def worker():
            try:
                # Phase 1: Get plan from planning LLM
                self.signals.log.emit("🧠 Generating action plan…", "info")
                plan_steps = generate_plan(self.planner, objective)

                if not plan_steps:
                    self.signals.log.emit("Planner returned empty plan. Falling back to direct mode.", "warn")
                    res = run_single_command(self.sandbox, self.llm, objective, self.signals, self.stop_event)
                    self.signals.finished.emit(f"Result: {res}")
                    return

                # Show plan in UI
                self.signals.plan_ready.emit(plan_steps)
                self.signals.log.emit(f"📋 Plan generated ({len(plan_steps)} steps):", "success")
                for i, step in enumerate(plan_steps, 1):
                    self.signals.log.emit(f"  {i}. {step}", "info")

                # Phase 2: Execute each step via Qwen3-VL
                def step_log(msg: str, level: str = "info"):
                    self.signals.log.emit(msg, level)

                res = run_planned_command(
                    sandbox=self.sandbox,
                    llm=self.llm,
                    plan_steps=plan_steps,
                    log=step_log,
                    stop_event=self.stop_event,
                )
                self.signals.finished.emit(f"Result: {res}")
            except Exception:
                self.signals.log.emit("ERROR:\n" + traceback.format_exc(), "error")
            finally:
                self.signals.busy.emit(False)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _run_direct(self, objective: str) -> None:
        """Original direct execution (no planner)."""
        def worker():
            try:
                res = run_single_command(
                    self.sandbox, self.llm, objective,
                    self.signals, self.stop_event)
                self.signals.finished.emit(f"Result: {res}")
            except Exception:
                self.signals.log.emit("ERROR:\n" + traceback.format_exc(), "error")
            finally:
                self.signals.busy.emit(False)

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _on_stop(self) -> None:
        if self.stop_event:
            self.stop_event.set()
            self.log_panel.append("Stop signal sent.", "warn")

    # --- Signal handlers ---
    def _on_log(self, msg: str, level: str) -> None:
        self.log_panel.append(msg, level)

    def _on_busy(self, busy: bool) -> None:
        self.cmd_panel.set_busy(busy)
        if self.vm_view:
            self.vm_view.input_enabled = not busy
        self.refresh_timer.setInterval(650 if busy else 350)

    def _on_finished(self, msg: str) -> None:
        elapsed = time.time() - self._run_start
        self.log_panel.append(msg, "success")
        self.inspector.set_metrics(self._step_count, self._click_count, self._type_count, elapsed)

    def _on_step(self, step_num: int, action: str, detail: str) -> None:
        self._step_count = step_num
        if action in ("CLICK", "DOUBLE_CLICK", "RIGHT_CLICK"):
            self._click_count += 1
        if action == "TYPE":
            self._type_count += 1
        self.cmd_panel.add_step(step_num, action, detail)
        self.top_bar.set_step(step_num)
        elapsed = time.time() - self._run_start
        self.inspector.set_metrics(self._step_count, self._click_count, self._type_count, elapsed)

    def _on_action(self, action_dict: dict) -> None:
        self.inspector.set_last_action(action_dict)

    def _on_latency(self, ms: float) -> None:
        self.top_bar.set_latency(ms)

    def _on_plan_ready(self, steps: list) -> None:
        self._current_plan = steps
        self.plan_display.set_plan(steps)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MissionControlWindowV2()
    w.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
