"""
gui.py — Графический интерфейс симулятора цифровой связи.
Python 3.12+

Изменения относительно предыдущей версии:
─────────────────────────────────────────────────────────────────────
Логирование:
  • Вынесено в _setup_logging() — больше нет дублирования между
    start_simulation() и run_simulation().
  • basicConfig больше не вызывается в simulation.py при импорте,
    поэтому здесь настройка единственная и корректная.

run_simulation():
  • log_config_once передаётся в simulate_transmission() / simulate_text_transmission()
    только на первой итерации — конфиг логируется ровно один раз за запуск.
  • Усреднение по num_simulations/num_repetitions вынесено в _average_stats(),
    убрано дублирование кода.
  • Текстовый файл загружается один раз (не в каждой итерации SNR).
  • Передаём execution_time в save_results_to_text().

process_queue() / _finish_simulation():
  • Состояние кнопок вынесено в _set_running_state().
  • stop_simulation() корректно сбрасывает состояние через queue.

build_config():
  • Валидация Eb/N0: start < stop, step > 0 — с понятным сообщением.
  • Валидация числовых полей с полным перечислением ошибок (не останавливается
    на первой).

show_history():
  • load_selected() использует try/except вокруг load_results() —
    если файл удалён вручную, окно не падает.

export_results():
  • PNG: явная проверка self._current_fig is not None.
  • JSON: использует _to_python() из results_manager для numpy-совместимости.

Дизайн:
  • Rician-канал добавлен в _build_channel_tab() (был в config, но не в GUI).
  • Mousewheel-scrolling в канал-табе теперь не конфликтует с основным окном:
    используется bind/unbind только внутри виджета (Enter/Leave).

Добавлена поддержка Reed-Solomon (март 2026):
  • На вкладке кодирования добавлена радиокнопка "Reed-Solomon (байтовый)".
  • При выборе RS отображается фрейм с параметрами: nsym, nsize, fcr, prim.
  • В build_config() добавлено считывание этих параметров в секцию coding.
"""

import csv
import json
import logging
import os
from pathlib import Path
import queue
import threading
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np

from simulation import (
    simulate_transmission,
    simulate_text_transmission,
    save_results_to_text,
    plot_and_save_results,
    plot_comparison,
)
from simulation_runner import results_manager
from results_manager import _to_python  # для JSON-экспорта

# ── Цветовая схема ────────────────────────────────────────────────────────────
DARK_BG       = "#0f0f23"
PANEL_BG      = "#16213e"
ACCENT_CYAN   = "#00d9ff"
ACCENT_PINK   = "#ff006e"
ACCENT_GREEN  = "#39ff14"
ACCENT_YELLOW = "#ffbe0b"
TEXT_COLOR    = "#e0e0e0"
TEXT_LIGHT    = "#ffffff"


# ── Вспомогательные функции GUI ───────────────────────────────────────────────

class ToolTip:
    """Простые подсказки при наведении (без внешних зависимостей)."""

    def __init__(self, widget: tk.Widget, text: str, *, delay_ms: int = 550) -> None:
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id: str | None = None
        self._tip: tk.Toplevel | None = None

        widget.bind("<Enter>", self._on_enter, add=True)
        widget.bind("<Leave>", self._on_leave, add=True)
        widget.bind("<ButtonPress>", self._on_leave, add=True)

    def _on_enter(self, _: object) -> None:
        if not self.text:
            return
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, _: object) -> None:
        self._cancel()
        self._hide()

    def _cancel(self) -> None:
        if self._after_id:
            try:
                self.widget.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _show(self) -> None:
        if self._tip or not self.widget.winfo_exists():
            return
        x = self.widget.winfo_rootx() + 18
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10

        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_attributes("-topmost", True)
        tip.configure(bg="#0b1220")

        label = tk.Label(
            tip,
            text=self.text,
            justify=tk.LEFT,
            bg="#0b1220",
            fg=TEXT_LIGHT,
            padx=10,
            pady=6,
            font=("Segoe UI", 9),
            wraplength=380,
        )
        label.pack()

        try:
            tip.wm_geometry(f"+{x}+{y}")
        except Exception:
            pass
        self._tip = tip

    def _hide(self) -> None:
        if self._tip:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def make_scrollable(tab: "ttk.Frame") -> "ttk.Frame":
    """
    Оборачивает содержимое вкладки в прокручиваемый контейнер.
    Возвращает inner_frame — в него помещается всё содержимое вкладки.
    Цвет фона совпадает с PANEL_BG (тёмно-синий).
    """
    import tkinter as _tk
    canvas = _tk.Canvas(tab, highlightthickness=0, bg=PANEL_BG)
    vsb = ttk.Scrollbar(tab, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=vsb.set)
    vsb.pack(side=_tk.RIGHT, fill=_tk.Y)
    canvas.pack(side=_tk.LEFT, fill=_tk.BOTH, expand=True)
    inner = ttk.Frame(canvas, style="TFrame")
    win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

    def _on_configure(e):
        canvas.configure(scrollregion=canvas.bbox("all"))
        canvas.itemconfig(win_id, width=canvas.winfo_width())
    inner.bind("<Configure>", _on_configure)
    canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))

    def _on_mousewheel(e):
        canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
    def _on_mousewheel_linux(e):
        canvas.yview_scroll(-1 if e.num == 4 else 1, "units")

    def _on_enter(_: object) -> None:
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        canvas.bind_all("<Button-4>", _on_mousewheel_linux)
        canvas.bind_all("<Button-5>", _on_mousewheel_linux)

    def _on_leave(_: object) -> None:
        canvas.unbind_all("<MouseWheel>")
        canvas.unbind_all("<Button-4>")
        canvas.unbind_all("<Button-5>")

    canvas.bind("<Enter>", _on_enter)
    canvas.bind("<Leave>", _on_leave)
    return inner


def make_section(parent: "ttk.Frame", title: str) -> "ttk.LabelFrame":
    """Создаёт LabelFrame с единым стилем отступов."""
    f = ttk.LabelFrame(parent, text=title, padding=10)
    f.pack(fill="x", padx=10, pady=6)
    return f


def add_row(frame: "ttk.Frame", row: int, label: str,
            widget_factory, **kw) -> "object":
    """Добавляет строку Label + виджет в grid-раскладку."""
    import tkinter as _tk
    ttk.Label(frame, text=label).grid(row=row, column=0, sticky=_tk.W, pady=4)
    w = widget_factory(**kw)
    w.grid(row=row, column=1, padx=8, sticky=_tk.W)
    return w



# ── Вспомогательная функция усреднения ───────────────────────────────────────

def _average_stats(stats_list: list[dict], snr: float) -> dict:
    """
    Усредняет список словарей статистики по числовым ключам.
    Нечисловые ключи (active_channels, text, ...) берутся из первого элемента.
    """
    if not stats_list:
        return {}
    keys = stats_list[0].keys()
    avg: dict = {"snr": snr}
    for k in keys:
        if k == "snr":
            continue
        vals = [s[k] for s in stats_list if isinstance(s.get(k), (int, float))]
        if vals:
            avg[k] = float(np.mean(vals))
        else:
            avg[k] = stats_list[0].get(k)
    return avg


def _flatten_result(r: dict) -> dict:
    """
    Разворачивает вложенные dict в плоскую структуру для CSV-экспорта.
    Ключи вида "parent_child". Списки и прочие сложные типы — str().
    """
    flat: dict = {}
    for k, v in r.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[f"{k}_{sk}"] = sv if not isinstance(sv, (dict, list)) else str(sv)
        elif isinstance(v, list):
            flat[k] = str(v)
        else:
            flat[k] = v
    return flat


def _auto_export_csv(results: list[dict], path: str) -> None:
    """Автоматический экспорт результатов в CSV после каждой симуляции."""
    flat = [_flatten_result(r) for r in results]
    if not flat:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
        writer.writeheader()
        writer.writerows(flat)


# ═════════════════════════════════════════════════════════════════════════════
class SimulationGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Симулятор цифровой связи — M-PSK / M-QAM")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 720)
        self.root.configure(bg=DARK_BG)
        try:
            self.root.iconbitmap(default="icon.ico")
        except Exception:
            pass

        self._setup_style()

        # --- Переменные состояния ---
        self.profile_var          = tk.StringVar(value="balanced")
        self.mode_var             = tk.StringVar(value="random")
        self.num_bits_var         = tk.StringVar(value="100000")
        self.num_simulations_var  = tk.StringVar(value="5")
        self.text_file_var        = tk.StringVar(value="input.txt")
        self.text_encoding_var    = tk.StringVar(value="utf-8")
        self.max_text_length_var  = tk.StringVar(value="10000")
        self.num_repetitions_var  = tk.StringVar(value="3")

        self.modulation_type_var  = tk.StringVar(value="PSK")
        self.modulation_order_var = tk.StringVar(value="8")
        self.gray_code_var        = tk.BooleanVar(value=True)

        self.coding_enabled_var   = tk.BooleanVar(value=True)
        self.coding_type_var      = tk.StringVar(value="hamming")
        self.turbo_iterations_var = tk.StringVar(value="6")
        self.turbo_block_size_var = tk.StringVar(value="128")

        # Переменные для Reed-Solomon
        self.rs_nsym_var          = tk.StringVar(value="10")
        self.rs_nsize_var         = tk.StringVar(value="255")
        self.rs_fcr_var           = tk.StringVar(value="0")
        self.rs_prim_var          = tk.StringVar(value="0x11d")

        self.ebn0_start_var       = tk.StringVar(value="0")
        self.ebn0_stop_var        = tk.StringVar(value="10")
        self.ebn0_step_var        = tk.StringVar(value="1")
        self.show_theo_var        = tk.BooleanVar(value=True)
        self.show_rayleigh_theo_var = tk.BooleanVar(value=True)

        # Адаптивный режим и ранняя остановка
        self.max_adaptive_bits_var = tk.StringVar(value="1000000")
        self.early_stop_ber_var    = tk.StringVar(value="1e-7")

        # PER настройки
        self.per_enabled_var  = tk.BooleanVar(value=False)
        self.per_packet_var   = tk.StringVar(value="1024")

        # Шифрование
        self.enc_enabled_var  = tk.BooleanVar(value=False)
        self.enc_type_var     = tk.StringVar(value="aes")
        self.enc_aes_mode_var = tk.StringVar(value="CBC")
        self.enc_key_var      = tk.StringVar(value="")

        # Перемежение
        self.il_enabled_var   = tk.BooleanVar(value=False)
        self.il_depth_var     = tk.StringVar(value="8")
        # Восстановление текста
        self.tr_enabled_var   = tk.BooleanVar(value=False)
        self.tr_window_var    = tk.StringVar(value="3")

        # Waveform (sps/RRC/CFO) — дискретно-временной режим
        self.wf_enabled_var     = tk.BooleanVar(value=False)
        self.wf_sps_var         = tk.StringVar(value="8")
        self.wf_rrc_beta_var    = tk.StringVar(value="0.35")
        self.wf_rrc_span_var    = tk.StringVar(value="10")
        self.wf_cfo_enabled_var = tk.BooleanVar(value=False)
        self.wf_cfo_norm_var    = tk.StringVar(value="0.0")
        self.wf_cfo_rec_var     = tk.BooleanVar(value=True)
        self.wf_pll_mode_var    = tk.StringVar(value="medium")
        self.wf_cfo_preamble_var = tk.BooleanVar(value=True)
        self.wf_cfo_prelen_var   = tk.StringVar(value="64")
        self.wf_cfo_loop_var     = tk.StringVar(value="dd_pll")
        self.wf_cfo_src_var      = tk.StringVar(value="preamble")
        # Waveform: TDL multipath + equalizer (symbol-rate)
        self.wf_tdl_enabled_var  = tk.BooleanVar(value=False)
        self.wf_tdl_delays_var   = tk.StringVar(value="0, 1, 3")
        self.wf_tdl_powers_var   = tk.StringVar(value="0, -3, -6")
        self.wf_tdl_seed_var     = tk.StringVar(value="123")
        self.wf_tdl_frac_taps_var = tk.StringVar(value="21")
        self.wf_eq_enabled_var   = tk.BooleanVar(value=False)
        self.wf_eq_kind_var      = tk.StringVar(value="mmse")
        self.wf_eq_len_var       = tk.StringVar(value="9")
        self.wf_eq_delay_var     = tk.StringVar(value="")
        self.wf_eq_chan_taps_var = tk.StringVar(value="32")
        self.wf_eq_est_var       = tk.StringVar(value="ideal")
        self.wf_eq_ls_reg_var    = tk.StringVar(value="1e-3")
        # Training + pilots for channel estimation (symbol-rate)
        self.wf_tr_enabled_var   = tk.BooleanVar(value=False)
        self.wf_tr_len_var       = tk.StringVar(value="128")
        self.wf_tr_seed_var      = tk.StringVar(value="12345")
        self.wf_pil_enabled_var  = tk.BooleanVar(value=False)
        self.wf_pil_period_var   = tk.StringVar(value="256")
        self.wf_pil_len_var      = tk.StringVar(value="16")
        self.wf_pil_seed_var     = tk.StringVar(value="4242")
        # Plots/metrics for waveform
        self.wf_plot_const_var   = tk.BooleanVar(value=True)
        self.wf_const_points_var = tk.StringVar(value="2000")
        self.wf_evm_chunk_var    = tk.StringVar(value="256")
        # Timing recovery (Gardner)
        self.wf_timing_enabled_var = tk.BooleanVar(value=False)
        self.wf_timing_alpha_var   = tk.StringVar(value="0.01")
        self.wf_timing_beta_var    = tk.StringVar(value="0.0001")
        self.wf_timing_init_var    = tk.StringVar(value="0.0")
        self.wf_timing_auto_var    = tk.BooleanVar(value=True)
        # Timing impairments (offset/drift/jitter) on samples
        self.wf_ti_enabled_var     = tk.BooleanVar(value=False)
        self.wf_ti_offset_var      = tk.StringVar(value="0.0")
        self.wf_ti_drift_ppm_var   = tk.StringVar(value="0.0")
        self.wf_ti_jitter_var      = tk.StringVar(value="0.0")
        self.wf_ti_seed_var        = tk.StringVar(value="2026")
        self.wf_ti_mode_var        = tk.StringVar(value="white")
        self.wf_ti_bw_var          = tk.StringVar(value="0.01")
        # Eye diagram
        self.wf_eye_enabled_var    = tk.BooleanVar(value=False)
        self.wf_eye_traces_var     = tk.StringVar(value="200")
        # Phase noise (samples)
        self.wf_pn_enabled_var     = tk.BooleanVar(value=False)
        self.wf_pn_step_deg_var    = tk.StringVar(value="0.2")
        self.wf_pn_seed_var        = tk.StringVar(value="777")
        # ADC / AGC / clipping / quantization
        self.wf_adc_enabled_var    = tk.BooleanVar(value=False)
        self.wf_dc_enabled_var     = tk.BooleanVar(value=False)
        self.wf_dc_i_var           = tk.StringVar(value="0.0")
        self.wf_dc_q_var           = tk.StringVar(value="0.0")
        self.wf_iq_enabled_var     = tk.BooleanVar(value=False)
        self.wf_iq_amp_db_var      = tk.StringVar(value="0.5")
        self.wf_iq_phase_deg_var   = tk.StringVar(value="3.0")
        self.wf_agc_enabled_var    = tk.BooleanVar(value=True)
        self.wf_agc_target_var     = tk.StringVar(value="1.0")
        self.wf_agc_min_gain_var   = tk.StringVar(value="0.01")
        self.wf_agc_max_gain_var   = tk.StringVar(value="100.0")
        self.wf_clip_enabled_var   = tk.BooleanVar(value=True)
        self.wf_clip_level_var     = tk.StringVar(value="1.2")
        self.wf_q_enabled_var      = tk.BooleanVar(value=True)
        self.wf_adc_bits_var       = tk.StringVar(value="10")
        self.wf_adc_fs_var         = tk.StringVar(value="1.0")

        self.queue           = queue.Queue()
        self.progress_var    = tk.DoubleVar(value=0.0)
        self.progress_label_var = tk.StringVar(value="Готов")
        self.eta_var         = tk.StringVar(value="")

        self.running         = False
        self.current_results = None
        self.current_config  = None
        self._current_canvas = None
        self._current_fig    = None

        self.channel_vars: dict[str, tk.Variable] = {}

        self.create_widgets()
        self.root.after(100, self.process_queue)
        self._bind_hotkeys()

        # Применяем профиль к UI-дефолтам после построения виджетов
        self._apply_profile_to_ui(self.profile_var.get())

    # ── Стиль ─────────────────────────────────────────────────────────────────

    def _setup_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",          background=PANEL_BG)
        style.configure("Dark.TFrame",     background=DARK_BG)
        style.configure("TLabel",          background=PANEL_BG, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Title.TLabel",    background=PANEL_BG, foreground=TEXT_LIGHT, font=("Segoe UI", 11, "bold"))
        style.configure("TButton",         font=("Segoe UI", 10, "bold"), padding=7)
        style.map("TButton", background=[("active", "#22304f"), ("pressed", "#243a66")])

        # Более “современные” акцентные/опасные кнопки
        style.configure("Accent.TButton",  background=ACCENT_CYAN, foreground="#000000")
        style.map("Accent.TButton",
                  background=[("active", "#4de7ff"), ("pressed", "#00b7d6"), ("disabled", "#2b4a5a")],
                  foreground=[("disabled", "#94a3b8")])
        style.configure("Danger.TButton",  background=ACCENT_PINK, foreground="#000000")
        style.map("Danger.TButton",
                  background=[("active", "#ff4f98"), ("pressed", "#d90060"), ("disabled", "#523041")],
                  foreground=[("disabled", "#94a3b8")])
        style.configure("Ghost.TButton",   background=PANEL_BG, foreground=TEXT_COLOR)
        style.map("Ghost.TButton",
                  background=[("active", "#1f2b4a"), ("pressed", "#22304f")])

        style.configure("TEntry",          fieldbackground="#1a1a2e", foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TCombobox",       fieldbackground="#1a1a2e", foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure(
            "Treeview",
            background="#1a1a2e",
            fieldbackground="#1a1a2e",
            foreground=TEXT_COLOR,
            font=("Segoe UI", 9),
            rowheight=26,
        )
        style.configure("Treeview.Heading", background=PANEL_BG, foreground=ACCENT_CYAN, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT_CYAN)], foreground=[("selected", "#000000")])
        style.configure("TPanedwindow", background=DARK_BG)
        style.configure("Sash", sashthickness=6)
        style.configure("TNotebook",       background=DARK_BG)
        style.configure("TNotebook.Tab",   background=PANEL_BG, foreground=TEXT_COLOR, font=("Segoe UI", 10, "bold"), padding=[15, 8])
        style.map("TNotebook.Tab",         background=[("selected", ACCENT_CYAN)], foreground=[("selected", "#000000")])
        style.configure("TCheckbutton",    background=PANEL_BG, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TRadiobutton",    background=PANEL_BG, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TLabelframe",     background=PANEL_BG, foreground=ACCENT_CYAN, font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background=PANEL_BG, foreground=ACCENT_CYAN)
        style.configure("TProgressbar",    background=ACCENT_CYAN, troughcolor="#1a1a2e")

    def _bind_hotkeys(self) -> None:
        # Горячие клавиши не должны ломать callbacks, которые не принимают event
        self.root.bind("<Control-Return>", lambda e: (not self.running) and self.start_simulation())
        self.root.bind("<Escape>",         lambda e: self.running and self.stop_simulation())
        self.root.bind("<Control-s>",      lambda e: self.save_results())
        self.root.bind("<Control-e>",      lambda e: self.export_results())
        self.root.bind("<Control-l>",      lambda e: self.show_log())

    # ── Логирование ───────────────────────────────────────────────────────────

    def _setup_logging(self) -> str:
        """
        Создаёт новый лог-файл для текущего запуска симуляции.
        Все предыдущие handlers удаляются, чтобы не писать в старый файл.
        Возвращает путь к созданному лог-файлу.
        """
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"sim_{timestamp}.log")

        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)

        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        fh = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        root_logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        root_logger.addHandler(ch)

        root_logger.setLevel(logging.INFO)
        root_logger.info(f"Лог-файл: {log_filename}")
        return log_filename

    # ── Состояние кнопок ──────────────────────────────────────────────────────

    def _set_running_state(self, running: bool) -> None:
        self.running = running
        self.start_btn.config(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL    if running else tk.DISABLED)

    def _toggle_section(self, widgets: list, var: "tk.BooleanVar") -> None:
        """Включает/выключает список виджетов по значению чекбокса."""
        state = tk.NORMAL if var.get() else tk.DISABLED
        for w in widgets:
            try:
                w.configure(state=state)
            except tk.TclError:
                pass

    # ── Виджеты ───────────────────────────────────────────────────────────────

    def create_widgets(self) -> None:
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL, style="TPanedwindow")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left = ttk.Frame(main, width=380, style="Dark.TFrame")
        right = ttk.Frame(main, style="Dark.TFrame")
        main.add(left, weight=0)
        main.add(right, weight=1)
        left.pack_propagate(False)

        # Заголовок
        hdr = ttk.Frame(left, style="Dark.TFrame")
        hdr.pack(fill=tk.X, padx=12, pady=(12, 8))
        ttk.Label(hdr, text="Симулятор цифровой связи", style="Title.TLabel",
                  foreground=TEXT_LIGHT).pack(anchor=tk.W)
        ttk.Label(hdr, text="Настройки и запуск моделирования", style="TLabel",
                  foreground="#94a3b8", background=DARK_BG, font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(2, 0))

        # Ноутбук с настройками
        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))

        tabs = [
            ("Данные",       self._build_data_tab),
            ("Кодирование",  self._build_coding_tab),
            ("Шифрование",   self._build_encryption_tab),
            ("Модуляция",    self._build_mod_tab),
            ("Канал",        self._build_channel_tab),
            ("Сигнал",       self._build_waveform_tab),
            ("Графики",      self._build_plot_tab),
            ("Перемежение",  self._build_interleaving_tab),
        ]
        for title, builder in tabs:
            tab_outer = ttk.Frame(nb)
            nb.add(tab_outer, text=title)
            inner = make_scrollable(tab_outer)
            builder(inner)

        # Кнопки управления
        btn_f = ttk.Frame(left)
        btn_f.pack(fill=tk.X, padx=10, pady=(0, 10))

        self.start_btn = ttk.Button(btn_f, text="▶ Старт", command=self.start_simulation, style="Accent.TButton")
        self.start_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(btn_f, text="⏹ Стоп", command=self.stop_simulation,
                                   state=tk.DISABLED, style="Danger.TButton")
        self.stop_btn.pack(fill=tk.X, pady=2)
        self.history_btn = ttk.Button(btn_f, text="История", command=self.show_history, style="Ghost.TButton")
        self.history_btn.pack(fill=tk.X, pady=2)
        self.logs_btn = ttk.Button(btn_f, text="Логи",    command=self.show_log, style="Ghost.TButton")
        self.logs_btn.pack(fill=tk.X, pady=2)

        ToolTip(self.start_btn, "Запуск симуляции.\nГорячая клавиша: Ctrl+Enter")
        ToolTip(self.stop_btn, "Остановить текущий запуск.\nГорячая клавиша: Esc")
        ToolTip(self.history_btn, "История запусков и загрузка сохранённых результатов.")
        ToolTip(self.logs_btn, "Открыть текущий лог.\nГорячая клавиша: Ctrl+L")

        # Правая часть
        ttk.Label(right, text="РЕЗУЛЬТАТЫ", style="Title.TLabel",
                  foreground=ACCENT_CYAN).pack(anchor=tk.W, pady=(0, 6), padx=6)
        self.plot_frame = ttk.Frame(right)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10), padx=6)

        ttk.Label(right, text="МЕТРИКИ", style="Title.TLabel",
                  foreground=ACCENT_CYAN).pack(anchor=tk.W, pady=(0, 6), padx=6)
        metrics_c = ttk.Frame(right, height=140)
        metrics_c.pack(fill=tk.X, side=tk.BOTTOM, padx=6, pady=(0, 4))
        metrics_c.pack_propagate(False)
        self._build_metrics_table(metrics_c)

        # Нижняя панель
        ctrl_f = ttk.Frame(self.root, style="Dark.TFrame")
        ctrl_f.pack(fill=tk.X, padx=10, pady=8)

        left_ctrl = ttk.Frame(ctrl_f, style="Dark.TFrame")
        left_ctrl.pack(side=tk.LEFT)
        ttk.Button(left_ctrl, text="Сохранить", command=self.save_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="Очистить",  command=self.clear_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="Экспорт",   command=self.export_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="CSV",       command=self.export_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="Сравнить",  command=self.compare_runs).pack(side=tk.LEFT, padx=4)

        right_ctrl = ttk.Frame(ctrl_f, style="Dark.TFrame")
        right_ctrl.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        prog_info = ttk.Frame(right_ctrl, style="Dark.TFrame")
        prog_info.pack(side=tk.LEFT, padx=20)
        ttk.Label(prog_info, textvariable=self.progress_label_var,
                  foreground=ACCENT_GREEN, style="Title.TLabel").pack(anchor=tk.W)
        ttk.Label(prog_info, textvariable=self.eta_var,
                  foreground=ACCENT_YELLOW, font=("Segoe UI", 9)).pack(anchor=tk.W)

        self.progress = ttk.Progressbar(right_ctrl, variable=self.progress_var,
                                        length=400, mode="determinate")
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    # ── Вкладки конфигурации ──────────────────────────────────────────────────

    def _build_data_tab(self, parent: ttk.Frame) -> None:
        # Профиль (presets)
        ttk.Label(parent, text="Профиль:", style="Title.TLabel").pack(
            anchor=tk.W, padx=10, pady=(10, 5)
        )
        pf = ttk.Frame(parent)
        pf.pack(anchor=tk.W, padx=10, pady=(0, 10))
        prof_cb = ttk.Combobox(
            pf,
            values=["fast", "balanced", "realistic"],
            textvariable=self.profile_var,
            width=14,
            state="readonly",
        )
        prof_cb.pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(
            pf,
            text="fast: быстро • balanced: реализм без перегруза • realistic: максимум блоков",
            foreground=ACCENT_YELLOW,
            font=("Segoe UI", 8),
        ).pack(side=tk.LEFT)

        def _on_profile_change(*_: object) -> None:
            self._apply_profile_to_ui(self.profile_var.get())

        self.profile_var.trace_add("write", _on_profile_change)

        ttk.Label(parent, text="Режим генерации:", style="Title.TLabel").pack(
            anchor=tk.W, padx=10, pady=(10, 5))
        mf = ttk.Frame(parent)
        mf.pack(anchor=tk.W, padx=10, pady=(0, 15))
        ttk.Radiobutton(mf, text="📊 Случайные биты",
                        variable=self.mode_var, value="random").pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(mf, text="📝 Текст",
                        variable=self.mode_var, value="text").pack(side=tk.LEFT, padx=8)

        rand_f = ttk.LabelFrame(parent, text="Параметры случайных бит", padding=10)
        rand_f.pack(fill=tk.X, padx=10, pady=8)
        for row, (label, var) in enumerate([
            ("Количество бит:",  self.num_bits_var),
            ("Число усреднений:", self.num_simulations_var),
        ]):
            ttk.Label(rand_f, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            ttk.Entry(rand_f, textvariable=var, width=15).grid(row=row, column=1, padx=10, sticky=tk.W)

        text_f = ttk.LabelFrame(parent, text="Параметры текста", padding=10)
        text_f.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(text_f, text="Файл:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(text_f, textvariable=self.text_file_var, width=15).grid(row=0, column=1, padx=10, sticky=tk.W)
        ttk.Button(text_f, text="📁", width=3, command=self.browse_file).grid(row=0, column=2, padx=5)

        ttk.Label(text_f, text="Кодировка:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(text_f, values=["utf-8", "utf-16", "ascii", "cp1251"],
                     textvariable=self.text_encoding_var, width=12, state="readonly").grid(
            row=1, column=1, padx=10, sticky=tk.W)

        for row, (label, var) in enumerate([
            ("Макс. длина (симв):", self.max_text_length_var),
            ("Повторений:",         self.num_repetitions_var),
        ], start=2):
            ttk.Label(text_f, text=label).grid(row=row, column=0, sticky=tk.W, pady=5)
            ttk.Entry(text_f, textvariable=var, width=15).grid(row=row, column=1, padx=10, sticky=tk.W)

        ttk.Label(parent, text="Диапазон Eb/N0 (дБ):", style="Title.TLabel").pack(
            anchor=tk.W, padx=10, pady=(15, 5))
        rf = ttk.Frame(parent)
        rf.pack(anchor=tk.W, padx=10, pady=(0, 10))
        for label, var in [("От:", self.ebn0_start_var), ("До:", self.ebn0_stop_var), ("Шаг:", self.ebn0_step_var)]:
            ttk.Label(rf, text=label).pack(side=tk.LEFT, padx=5)
            ttk.Entry(rf, textvariable=var, width=6).pack(side=tk.LEFT, padx=5)

        # Адаптивный режим и ранняя остановка
        self.adv_enabled_var = tk.BooleanVar(value=True)
        adv_outer = ttk.LabelFrame(parent, text="Дополнительно", padding=10)
        adv_outer.pack(fill=tk.X, padx=10, pady=8)
        ttk.Checkbutton(adv_outer, text="Включить",
                        variable=self.adv_enabled_var,
                        command=lambda: self._toggle_section(adv_widgets, self.adv_enabled_var)
                        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 4))
        adv_f = ttk.Frame(adv_outer)
        adv_f.grid(row=1, column=0, columnspan=3, sticky=tk.EW)
        adv_widgets: list = []
        for row, (label, var, tip) in enumerate([
            ("Макс. бит (адаптивно):", self.max_adaptive_bits_var, "×10 при BER<1e-4, ×100 при BER<1e-5"),
            ("Ранняя остановка BER:", self.early_stop_ber_var,     "0 = отключено; напр. 1e-7"),
        ]):
            lbl = ttk.Label(adv_f, text=label)
            lbl.grid(row=row, column=0, sticky=tk.W, pady=4)
            ent = ttk.Entry(adv_f, textvariable=var, width=14)
            ent.grid(row=row, column=1, padx=8, sticky=tk.W)
            tip_lbl = ttk.Label(adv_f, text=tip, foreground=ACCENT_YELLOW,
                      font=("Segoe UI", 8))
            tip_lbl.grid(row=row, column=2, sticky=tk.W, padx=4)
            adv_widgets.extend([lbl, ent, tip_lbl])

        # PER настройки
        per_f = ttk.LabelFrame(parent, text="PER (Packet Error Rate)", padding=10)
        per_f.pack(fill=tk.X, padx=10, pady=8)
        ttk.Checkbutton(per_f, text="✓ Вычислять PER",
                        variable=self.per_enabled_var).grid(row=0, column=0, columnspan=3,
                                                             sticky=tk.W, pady=(0, 6))
        ttk.Label(per_f, text="Размер пакета (бит):").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(per_f, textvariable=self.per_packet_var, width=10).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(per_f, text="напр. 1024", foreground=ACCENT_YELLOW,
                  font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)

    def _apply_profile_to_ui(self, profile: str) -> None:
        """
        Проставляет дефолты «как в системе» в UI.
        Важно: это именно пресет для удобства; пользователь всегда может потом переопределить руками.
        """
        p = (profile or "fast").lower().strip()
        if p == "fast":
            self.wf_enabled_var.set(False)
            self.wf_cfo_enabled_var.set(False)
            self.wf_timing_enabled_var.set(False)
            return

        if p == "balanced":
            self.wf_enabled_var.set(True)
            self.wf_sps_var.set("8")
            self.wf_rrc_beta_var.set("0.35")
            self.wf_rrc_span_var.set("10")

            self.wf_cfo_enabled_var.set(True)
            self.wf_cfo_norm_var.set("0.0")
            self.wf_cfo_rec_var.set(True)
            self.wf_cfo_preamble_var.set(True)
            self.wf_cfo_prelen_var.set("64")
            self.wf_cfo_loop_var.set("dd_pll")
            self.wf_cfo_src_var.set("preamble")

            self.wf_timing_enabled_var.set(True)
            self.wf_timing_alpha_var.set("0.01")
            self.wf_timing_beta_var.set("0.0001")
            self.wf_timing_init_var.set("0.0")
            self.wf_timing_auto_var.set(True)

            # Balanced: тяжёлые/дорогие блоки оставляем выключенными
            self.wf_tdl_enabled_var.set(False)
            self.wf_eq_enabled_var.set(False)
            self.wf_tr_enabled_var.set(False)
            self.wf_pil_enabled_var.set(False)
            self.wf_adc_enabled_var.set(False)
            self.wf_pn_enabled_var.set(False)
            self.wf_ti_enabled_var.set(False)
            self.wf_eye_enabled_var.set(False)
            return

        if p == "realistic":
            self.wf_enabled_var.set(True)
            self.wf_cfo_enabled_var.set(True)
            self.wf_cfo_norm_var.set("0.0005")
            self.wf_cfo_rec_var.set(True)
            self.wf_timing_enabled_var.set(True)
            self.wf_tdl_enabled_var.set(True)
            self.wf_eq_enabled_var.set(True)
            self.wf_pil_enabled_var.set(True)
            self.wf_adc_enabled_var.set(True)
            self.wf_pn_enabled_var.set(True)
            # В realistic можно оставить глаз выключенным, чтобы не тормозить UI
            self.wf_eye_enabled_var.set(False)
            return

    def _build_coding_tab(self, parent: ttk.Frame) -> None:
        ttk.Checkbutton(parent, text="✓ Включить кодирование",
                        variable=self.coding_enabled_var).pack(anchor=tk.W, padx=15, pady=10)
        ttk.Label(parent, text="Тип кодирования:", style="Title.TLabel").pack(
            anchor=tk.W, padx=15, pady=(10, 5))
        cf = ttk.Frame(parent)
        cf.pack(anchor=tk.W, padx=15, pady=(0, 8))
        ttk.Radiobutton(cf, text="🔷 Hamming (7,4)",  variable=self.coding_type_var,
                        value="hamming").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(cf, text="🔸 Turbo (PCCC≈1/3)", variable=self.coding_type_var,
                        value="turbo").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(cf, text="🔹 Reed-Solomon (байтовый)", variable=self.coding_type_var,
                        value="reed-solomon").pack(anchor=tk.W, pady=2)

        # Параметры Turbo (показываются только при выборе Turbo)
        turbo_f = ttk.LabelFrame(parent, text="Параметры Turbo", padding=8)
        for row, (label, var) in enumerate([
            ("Итераций декодера:", self.turbo_iterations_var),
            ("Размер блока (бит):", self.turbo_block_size_var),
        ]):
            ttk.Label(turbo_f, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
            ttk.Entry(turbo_f, textvariable=var, width=10).grid(row=row, column=1, padx=8, sticky=tk.W)

        # Параметры Reed-Solomon
        rs_f = ttk.LabelFrame(parent, text="Параметры Reed-Solomon", padding=8)
        for row, (label, var, default) in enumerate([
            ("ECC символов (nsym):", self.rs_nsym_var, "10"),
            ("Размер блока (nsize):", self.rs_nsize_var, "255"),
            ("Первый корень (fcr):", self.rs_fcr_var, "0"),
            ("Примитивный полином (hex):", self.rs_prim_var, "0x11d"),
        ]):
            ttk.Label(rs_f, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
            entry = ttk.Entry(rs_f, textvariable=var, width=12)
            entry.grid(row=row, column=1, padx=8, sticky=tk.W)

        def _toggle_frames(*_: object) -> None:
            # Скрываем оба фрейма
            turbo_f.pack_forget()
            rs_f.pack_forget()
            if self.coding_type_var.get() == "turbo":
                turbo_f.pack(fill=tk.X, padx=15, pady=(0, 6))
            elif self.coding_type_var.get() == "reed-solomon":
                rs_f.pack(fill=tk.X, padx=15, pady=(0, 6))

        self.coding_type_var.trace_add("write", _toggle_frames)
        _toggle_frames()  # изначально скрыты (default = hamming)

        ttk.Label(parent,
                  text=("Hamming (7,4):\n"
                        "  • Быстрое декодирование\n"
                        "  • Исправляет 1 ошибку на блок\n"
                        "  • Скорость 4/7 ≈ 0.57\n\n"
                        "Turbo (PCCC, Log-MAP):\n"
                        "  • Лучшая помехоустойчивость\n"
                        "  • Скорость ≈ 1/3 (медленнее)\n\n"
                        "Reed-Solomon:\n"
                        "  • Байтовый код (исправляет пакеты ошибок)\n"
                        "  • Исправляет до floor(nsym/2) ошибок\n"
                        "  • nsize ≤ 255 для GF(2^8)"),
                  justify=tk.LEFT, foreground=ACCENT_YELLOW).pack(anchor=tk.W, padx=15, pady=8)

    def _build_encryption_tab(self, parent: ttk.Frame) -> None:
        """Вкладка настроек шифрования."""
        import os as _os
        from encryption import aes_available as _aes_ok

        # ── Включить / выключить ─────────────────────────────────────────────
        ttk.Checkbutton(parent, text="✓ Включить шифрование",
                        variable=self.enc_enabled_var).pack(
            anchor=tk.W, padx=15, pady=(12, 6))

        # ── Алгоритм ─────────────────────────────────────────────────────────
        alg_f = ttk.LabelFrame(parent, text="Алгоритм", padding=8)
        alg_f.pack(fill=tk.X, padx=15, pady=(0, 8))

        aes_rb = ttk.Radiobutton(
            alg_f,
            text="🔒 AES-128 (стандарт)" + ("" if _aes_ok() else "  ⚠ недоступен"),
            variable=self.enc_type_var, value="aes",
            state=tk.NORMAL if _aes_ok() else tk.DISABLED,
        )
        aes_rb.pack(anchor=tk.W, pady=2)

        # ── Режим AES ────────────────────────────────────────────────────────
        mode_f = ttk.LabelFrame(parent, text="Режим AES", padding=8)
        mode_f.pack(fill=tk.X, padx=15, pady=(0, 8))

        modes_info = [
            ("CBC", "CBC  — цепочка, ошибка разрушает 1 блок + 1 бит следующего"),
            ("CTR", "CTR  — потоковый режим, нет распространения ошибок"),
        ]
        for val, text in modes_info:
            ttk.Radiobutton(mode_f, text=text,
                            variable=self.enc_aes_mode_var, value=val).pack(
                anchor=tk.W, pady=2)

        def _toggle_mode_frame(*_):
            if self.enc_type_var.get() == "aes":
                mode_f.pack(fill=tk.X, padx=15, pady=(0, 8))
            else:
                mode_f.pack_forget()

        self.enc_type_var.trace_add("write", _toggle_mode_frame)
        _toggle_mode_frame()

        # ── Ключ ─────────────────────────────────────────────────────────────
        key_f = ttk.LabelFrame(parent, text="Ключ шифрования", padding=8)
        key_f.pack(fill=tk.X, padx=15, pady=(0, 8))

        ttk.Label(key_f, text="Hex (32 символа = 128 бит):").grid(
            row=0, column=0, sticky=tk.W, pady=(0, 4))

        key_entry = ttk.Entry(key_f, textvariable=self.enc_key_var, width=34,
                              font=("Courier", 9))
        key_entry.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 6))
        key_f.columnconfigure(0, weight=1)

        def _gen_key():
            self.enc_key_var.set(_os.urandom(16).hex())

        ttk.Button(key_f, text="🎲 Случайный ключ", command=_gen_key).grid(
            row=2, column=0, sticky=tk.W)

        ttk.Label(key_f,
                  text="Пусто → новый случайный ключ на каждый прогон",
                  foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

        # ── Справка ──────────────────────────────────────────────────────────
        ttk.Label(parent,
                  text=("Шифрование применяется ДО кодирования:\n"
                        "  данные → шифр → кодер → канал → декодер → дешифр\n\n"
                        "Метрики после включения:\n"
                        "  BER pre-decrypt  — ошибки канала (до дешифровки)\n"
                        "  BER post-decrypt — итоговый BER (основной)\n"
                        "  Распр. ошибок    — BER_post / BER_pre\n"
                        "  AES блок-ошибки  — повреждённых 128-бит блоков\n\n"
                        "CTR: распр. ≈ 1.0\n"
                        "CBC: распр. > 1.0 при средних SNR (лавинный эффект)"),
                  justify=tk.LEFT,
                  foreground=ACCENT_YELLOW).pack(anchor=tk.W, padx=15, pady=8)

    def _build_mod_tab(self, parent: ttk.Frame) -> None:
        ttk.Label(parent, text="Тип модуляции:", style="Title.TLabel").pack(
            anchor=tk.W, padx=15, pady=(10, 5))
        mf = ttk.Frame(parent)
        mf.pack(anchor=tk.W, padx=15, pady=(0, 20))
        ttk.Radiobutton(mf, text="📡 PSK", variable=self.modulation_type_var,
                        value="PSK").pack(side=tk.LEFT, padx=10, pady=5)
        ttk.Radiobutton(mf, text="📡 QAM", variable=self.modulation_type_var,
                        value="QAM").pack(side=tk.LEFT, padx=10, pady=5)

        ttk.Label(parent, text="Порядок модуляции (M):", style="Title.TLabel").pack(
            anchor=tk.W, padx=15, pady=(10, 5))
        self.mod_order_combo = ttk.Combobox(
            parent, textvariable=self.modulation_order_var,
            values=["2", "4", "8", "16"], state="readonly", width=10,
            font=("Segoe UI", 10, "bold"))
        self.mod_order_combo.pack(anchor=tk.W, padx=15, pady=(0, 20))

        def _update_mod_order(*_: object) -> None:
            if self.modulation_type_var.get() == "QAM":
                self.mod_order_combo["values"] = ("4", "16", "64", "256")
                if self.modulation_order_var.get() not in ("4", "16", "64", "256"):
                    self.modulation_order_var.set("16")
            else:
                self.mod_order_combo["values"] = ("2", "4", "8", "16")
                if self.modulation_order_var.get() not in ("2", "4", "8", "16"):
                    self.modulation_order_var.set("8")

        self.modulation_type_var.trace_add("write", _update_mod_order)
        ttk.Checkbutton(parent, text="✓ Код Грея",
                        variable=self.gray_code_var).pack(anchor=tk.W, padx=15, pady=(0, 20))
        ttk.Label(parent,
                  text=("PSK: фазовая манипуляция\n"
                        "  • M = 2, 4, 8, 16\n\n"
                        "QAM: амплитудно-фазовая\n"
                        "  • M = 4, 16, 64, 256\n"
                        "  • Выше спектр. эффективность\n"
                        "  • QAM-256: 8 бит/символ"),
                  justify=tk.LEFT, foreground=ACCENT_YELLOW).pack(anchor=tk.W, padx=15, pady=10)

    def _build_channel_tab(self, parent: ttk.Frame) -> None:
        # parent уже является прокручиваемым inner_frame от make_scrollable
        sf = parent

        # AWGN — всегда
        af = ttk.LabelFrame(sf, text="🌊 AWGN (Белый шум)", padding=10)
        af.pack(fill=tk.X, padx=10, pady=8)
        ttk.Label(af, text="Добавляется автоматически через Eb/N0",
                  foreground=ACCENT_YELLOW).pack(anchor=tk.W)

        def _add_section(title: str, key_prefix: str, fields: list[tuple]) -> tuple[dict, ttk.LabelFrame]:
            """Создаёт LabelFrame с чекбоксом-включателем и набором полей.
            Возвращает (vars_map, frame)."""
            enabled_var = tk.BooleanVar(value=False)
            frame = ttk.LabelFrame(sf, text=title, padding=10)
            frame.pack(fill=tk.X, padx=10, pady=8)
            ttk.Checkbutton(frame, text=f"✓ Включить", variable=enabled_var).pack(
                anchor=tk.W, pady=(0, 8))
            vars_map: dict[str, tk.Variable] = {f"{key_prefix}_enabled": enabled_var}
            for label, var_key, var_type, default, widget_kwargs in fields:
                ttk.Label(frame, text=label, font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(4, 0))
                # В GUI храним всё как строки; преобразование делаем в build_config().
                var = tk.StringVar(value=str(default))
                widget_class = widget_kwargs.pop("__class__", ttk.Entry)
                w = widget_class(frame, textvariable=var, **widget_kwargs)
                w.pack(anchor=tk.W, pady=(0, 6))
                vars_map[f"{key_prefix}_{var_key}"] = var
            return vars_map, frame

        # Rayleigh
        _vars, _ = _add_section(
            "📶 Rayleigh Fading (многолучевое)", "rayleigh", [
                ("Число лучей:",              "rays",    "str", "16",  {"__class__": ttk.Spinbox, "from_": 2, "to": 64, "width": 10}),
                ("Доплер. частота (норм.):",  "doppler", "str", "0.01", {"width": 15}),
            ]
        )
        self.channel_vars.update(_vars)

        # Phase Noise
        _vars, _ = _add_section(
            "🔄 Фазовый шум", "phase", [
                # В channel.py ожидается std в градусах приращения фазы (phase_noise_std_deg).
                ("СКО приращения фазы (град):", "std_deg", "str", "1.0", {"width": 15}),
            ]
        )
        self.channel_vars.update(_vars)

        # Impulse Noise (гибко: одиночные / случайные пачки / РЛС-пачки)
        _vars, _ = _add_section(
            "⚡ Импульсные помехи", "impulse_noise", [
                ("Режим:", "mode", "str", "bernoulli", {
                    "__class__": ttk.Combobox,
                    "values": ("bernoulli", "random_bursts", "radar"),
                    "state": "readonly",
                    "width": 15,
                }),
                ("SNR импульса (дБ):", "snr_db", "str", "20.0", {"width": 15}),
                # bernoulli
                ("Вероятность импульса (p):", "prob", "str", "0.001", {"width": 15}),
                # random_bursts
                ("P старта пачки:", "burst_p", "str", "0.0001", {"width": 15}),
                ("Длина пачки min:", "burst_len_min", "str", "4", {"__class__": ttk.Spinbox, "from_": 1, "to": 100000, "width": 10}),
                ("Длина пачки max:", "burst_len_max", "str", "32", {"__class__": ttk.Spinbox, "from_": 1, "to": 100000, "width": 10}),
                ("Пауза min (симв):", "gap_min", "str", "32", {"__class__": ttk.Spinbox, "from_": 0, "to": 100000, "width": 10}),
                ("Пауза max (симв):", "gap_max", "str", "256", {"__class__": ttk.Spinbox, "from_": 0, "to": 100000, "width": 10}),
                ("P импульса внутри пачки:", "inburst_p", "str", "1.0", {"width": 15}),
                ("Ширина импульса (симв):", "width", "str", "1", {"__class__": ttk.Spinbox, "from_": 1, "to": 100000, "width": 10}),
                # radar
                ("Период пачки (симв):", "radar_period", "str", "2000", {"__class__": ttk.Spinbox, "from_": 1, "to": 10000000, "width": 10}),
                ("Импульсов в пачке:", "radar_pulses", "str", "16", {"__class__": ttk.Spinbox, "from_": 1, "to": 100000, "width": 10}),
                ("Шаг импульса (симв):", "radar_spacing", "str", "20", {"__class__": ttk.Spinbox, "from_": 1, "to": 100000, "width": 10}),
                ("Ширина импульса (симв):", "radar_width", "str", "1", {"__class__": ttk.Spinbox, "from_": 1, "to": 100000, "width": 10}),
                ("Джиттер (±симв):", "radar_jitter", "str", "0", {"__class__": ttk.Spinbox, "from_": 0, "to": 100000, "width": 10}),
            ]
        )
        self.channel_vars.update(_vars)


    def _build_interleaving_tab(self, parent: ttk.Frame) -> None:
        """Вкладка «Перемежение и восстановление текста»."""
        # ── Интерливер ──────────────────────────────────────────────────────
        il_f = ttk.LabelFrame(parent, text="🔀 Блочный интерливер", padding=10)
        il_f.pack(fill=tk.X, padx=10, pady=(12, 8))

        ttk.Checkbutton(
            il_f, text="✓ Включить перемежение",
            variable=self.il_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        ttk.Label(il_f, text="Глубина (depth):").grid(row=1, column=0, sticky=tk.W, pady=4)
        depth_cb = ttk.Combobox(
            il_f, textvariable=self.il_depth_var,
            values=["4", "8", "16", "32"], state="readonly", width=8,
        )
        depth_cb.grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(il_f, text="бит", foreground=ACCENT_YELLOW,
                  font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)

        ttk.Label(
            il_f,
            text=(
                "Матрица записывается по строкам,\n"
                "читается по столбцам.\n"
                "Глубина 8 → пакет до 8 бит рассыпается\n"
                "в одиночные ошибки после деперемежения."
            ),
            foreground=ACCENT_YELLOW,
            justify=tk.LEFT,
            font=("Segoe UI", 8),
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))

        # ── Восстановление текста ────────────────────────────────────────────
        tr_f = ttk.LabelFrame(parent, text="🔧 Восстановление текста (UTF-8 repair)", padding=10)
        tr_f.pack(fill=tk.X, padx=10, pady=(0, 8))

        ttk.Checkbutton(
            tr_f, text="✓ Включить восстановление",
            variable=self.tr_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        ttk.Label(tr_f, text="Окно поиска (байт):").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(
            tr_f, from_=1, to=8, textvariable=self.tr_window_var, width=6,
        ).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(tr_f, text="≥1", foreground=ACCENT_YELLOW,
                  font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)

        ttk.Label(
            tr_f,
            text=(
                "Для каждого нечитаемого символа (U+FFFD)\n"
                "перебирает битовые сдвиги в окне вокруг\n"
                "повреждённого байта. Выбирает вариант\n"
                "с наибольшим числом читаемых символов.\n\n"
                "Эффективно при BER < 5% и кириллице."
            ),
            foreground=ACCENT_YELLOW,
            justify=tk.LEFT,
            font=("Segoe UI", 8),
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(6, 0))

        # ── Кнопка панели восстановления ────────────────────────────────────
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=10, pady=8)
        ttk.Button(
            parent,
            text="📋 Открыть панель восстановления текста",
            command=self._open_recovery_panel,
        ).pack(padx=10, pady=4, fill=tk.X)

        ttk.Label(
            parent,
            text=(
                "Панель показывает оригинал и восстановленный\n"
                "текст рядом. Повреждённые символы выделены."
            ),
            foreground=ACCENT_YELLOW,
            justify=tk.LEFT,
            font=("Segoe UI", 8),
        ).pack(padx=10, pady=(2, 0), anchor=tk.W)

    def _build_waveform_tab(self, parent: ttk.Frame) -> None:
        """
        Вкладка «Сигнал»: дискретно-временная модель (sps>1, RRC, CFO).

        Важно:
          - При включении waveform-режима канал из вкладки «Канал» пока
            применяется только в symbol-rate режиме (старом).
          - В waveform-режиме текущий MVP: RRC(TX/RX) + AWGN (+CFO).
        """
        wf_f = ttk.LabelFrame(parent, text="📶 Waveform (sps + RRC)", padding=10)
        wf_f.pack(fill=tk.X, padx=10, pady=(12, 8))

        ttk.Checkbutton(
            wf_f, text="✓ Включить дискретно-временной режим (sps>1)",
            variable=self.wf_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        ttk.Label(wf_f, text="Отсчётов на символ (sps):").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            wf_f, textvariable=self.wf_sps_var,
            values=["2", "4", "8", "16"], state="readonly", width=8,
        ).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(wf_f, text=">=2", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(
            row=1, column=2, sticky=tk.W
        )

        ttk.Label(wf_f, text="RRC roll-off (beta):").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(wf_f, textvariable=self.wf_rrc_beta_var, width=10).grid(
            row=2, column=1, padx=8, sticky=tk.W
        )
        ttk.Label(wf_f, text="0..1 (напр. 0.35)", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(
            row=2, column=2, sticky=tk.W
        )

        ttk.Label(wf_f, text="RRC длина (span, символов):").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(
            wf_f, from_=4, to=20, textvariable=self.wf_rrc_span_var, width=8,
        ).grid(row=3, column=1, padx=8, sticky=tk.W)
        ttk.Label(wf_f, text="типично 6..12", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(
            row=3, column=2, sticky=tk.W
        )

        cfo_f = ttk.LabelFrame(parent, text="🎛 CFO (carrier offset)", padding=10)
        cfo_f.pack(fill=tk.X, padx=10, pady=(0, 8))

        ttk.Checkbutton(
            cfo_f, text="✓ Включить CFO",
            variable=self.wf_cfo_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        ttk.Label(cfo_f, text="CFO norm (циклов/отсчёт):").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(cfo_f, textvariable=self.wf_cfo_norm_var, width=12).grid(
            row=1, column=1, padx=8, sticky=tk.W
        )
        ttk.Label(
            cfo_f,
            text="напр. 0.001 (малое вращение)",
            foreground=ACCENT_YELLOW, font=("Segoe UI", 8),
        ).grid(row=1, column=2, sticky=tk.W)

        ttk.Checkbutton(
            cfo_f, text="✓ CFO recovery (PLL, decision-directed)",
            variable=self.wf_cfo_rec_var,
        ).grid(row=2, column=0, columnspan=3, sticky=tk.W, pady=(6, 4))

        ttk.Label(cfo_f, text="Тип петли:").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            cfo_f, textvariable=self.wf_cfo_loop_var,
            values=["dd_pll", "costas_mpsk"], state="readonly", width=12,
        ).grid(row=3, column=1, padx=8, sticky=tk.W)
        ttk.Label(
            cfo_f, text="PSK: можно выбрать Costas",
            foreground=ACCENT_YELLOW, font=("Segoe UI", 8),
        ).grid(row=3, column=2, sticky=tk.W)

        ttk.Label(cfo_f, text="Скорость петли (PLL):").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            cfo_f, textvariable=self.wf_pll_mode_var,
            values=["slow", "medium", "fast"], state="readonly", width=10,
        ).grid(row=4, column=1, padx=8, sticky=tk.W)
        ttk.Label(
            cfo_f, text="slow=стабильнее, fast=быстрее",
            foreground=ACCENT_YELLOW, font=("Segoe UI", 8),
        ).grid(row=4, column=2, sticky=tk.W)

        ttk.Checkbutton(
            cfo_f, text="✓ Переамбула для CFO (2 одинаковые половины)",
            variable=self.wf_cfo_preamble_var,
        ).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(10, 4))

        ttk.Label(cfo_f, text="Длина половины (символов):").grid(row=6, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(
            cfo_f, from_=16, to=512, increment=16,
            textvariable=self.wf_cfo_prelen_var, width=8,
        ).grid(row=6, column=1, padx=8, sticky=tk.W)
        ttk.Label(
            cfo_f, text="типично 32..128",
            foreground=ACCENT_YELLOW, font=("Segoe UI", 8),
        ).grid(row=6, column=2, sticky=tk.W)

        ttk.Label(cfo_f, text="Estimator source:").grid(row=7, column=0, sticky=tk.W, pady=(8, 4))
        ttk.Combobox(
            cfo_f, textvariable=self.wf_cfo_src_var,
            values=["training", "pilots", "pilots_track", "pilots_track_lin", "preamble", "blind"], state="readonly", width=12,
        ).grid(row=7, column=1, padx=8, sticky=tk.W, pady=(8, 4))
        ttk.Label(
            cfo_f, text="выбор источника оценки CFO",
            foreground=ACCENT_YELLOW, font=("Segoe UI", 8),
        ).grid(row=7, column=2, sticky=tk.W, pady=(8, 4))

        ttk.Label(
            parent,
            text=(
                "Waveform-режим добавляет >1 отсчёта/символ и фильтры как в реальном PHY:\n"
                "  symbols → upsample → RRC(TX) → [CFO] → AWGN → RRC(RX) → downsample → demod\n\n"
                "Пока это MVP: без тайминг-синхронизации. Можно добавить TDL-многолучевой канал и эквалайзер.\n"
                "Старый режим (symbol-rate) остаётся доступен при выключенном чекбоксе."
            ),
            justify=tk.LEFT, foreground=ACCENT_YELLOW,
        ).pack(anchor=tk.W, padx=10, pady=10)

        tdl_f = ttk.LabelFrame(parent, text="🌁 TDL многолучевой (на отсчётах)", padding=10)
        tdl_f.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Checkbutton(
            tdl_f, text="✓ Включить TDL (свёртка FIR на отсчётах)",
            variable=self.wf_tdl_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        ttk.Label(tdl_f, text="Задержки (символы):").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(tdl_f, textvariable=self.wf_tdl_delays_var, width=18).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(tdl_f, text="напр. 0,1,3", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)

        ttk.Label(tdl_f, text="Мощности путей (дБ):").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(tdl_f, textvariable=self.wf_tdl_powers_var, width=18).grid(row=2, column=1, padx=8, sticky=tk.W)
        ttk.Label(tdl_f, text="напр. 0,-3,-6", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=2, column=2, sticky=tk.W)

        ttk.Label(tdl_f, text="Seed (пусто = случайно):").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Entry(tdl_f, textvariable=self.wf_tdl_seed_var, width=10).grid(row=3, column=1, padx=8, sticky=tk.W)
        ttk.Label(tdl_f, text="Дробн. задержки (frac taps):").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(tdl_f, from_=5, to=101, increment=2, textvariable=self.wf_tdl_frac_taps_var, width=8).grid(
            row=4, column=1, padx=8, sticky=tk.W
        )
        ttk.Label(tdl_f, text="нечётное (напр. 21)", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(
            row=4, column=2, sticky=tk.W
        )

        eq_f = ttk.LabelFrame(parent, text="🧰 Эквалайзер (symbol-rate, без пилотов)", padding=10)
        eq_f.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Checkbutton(
            eq_f, text="✓ Включить линейный эквалайзер",
            variable=self.wf_eq_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        ttk.Label(eq_f, text="Тип:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            eq_f, textvariable=self.wf_eq_kind_var,
            values=["zf", "mmse"], state="readonly", width=10,
        ).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(eq_f, text="MMSE устойчивее при шуме", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)

        ttk.Label(eq_f, text="Оценка канала:").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            eq_f, textvariable=self.wf_eq_est_var,
            values=["ideal", "training_ls", "pilot_ls", "preamble_ls"], state="readonly", width=12,
        ).grid(row=2, column=1, padx=8, sticky=tk.W)
        ttk.Label(eq_f, text="training/pilot — отдельные настройки ниже", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(
            row=2, column=2, sticky=tk.W
        )

        ttk.Label(eq_f, text="Длина FIR (taps):").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(eq_f, from_=3, to=63, increment=2, textvariable=self.wf_eq_len_var, width=8).grid(row=2, column=1, padx=8, sticky=tk.W)

        ttk.Label(eq_f, text="Delay (пусто = авто):").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Entry(eq_f, textvariable=self.wf_eq_delay_var, width=10).grid(row=3, column=1, padx=8, sticky=tk.W)
        ttk.Label(eq_f, text="обычно Lh-1", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=3, column=2, sticky=tk.W)

        ttk.Label(eq_f, text="Оценка taps канала (символов):").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(eq_f, from_=16, to=128, increment=8, textvariable=self.wf_eq_chan_taps_var, width=8).grid(row=4, column=1, padx=8, sticky=tk.W)

        ttk.Label(eq_f, text="LS reg (для *_ls):").grid(row=5, column=0, sticky=tk.W, pady=4)
        ttk.Entry(eq_f, textvariable=self.wf_eq_ls_reg_var, width=10).grid(row=5, column=1, padx=8, sticky=tk.W)
        ttk.Label(eq_f, text="напр. 1e-3", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=5, column=2, sticky=tk.W)

        tr_f = ttk.LabelFrame(parent, text="🎯 Training sequence (оценка канала)", padding=10)
        tr_f.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Checkbutton(
            tr_f, text="✓ Включить training в начале кадра",
            variable=self.wf_tr_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(tr_f, text="Длина (символов):").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(tr_f, from_=16, to=2048, increment=16, textvariable=self.wf_tr_len_var, width=8).grid(
            row=1, column=1, padx=8, sticky=tk.W
        )
        ttk.Label(tr_f, text="типично 64..256", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)
        ttk.Label(tr_f, text="Seed:").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(tr_f, textvariable=self.wf_tr_seed_var, width=12).grid(row=2, column=1, padx=8, sticky=tk.W)

        pil_f = ttk.LabelFrame(parent, text="🧷 Пилоты в данных (каждые N символов)", padding=10)
        pil_f.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Checkbutton(
            pil_f, text="✓ Вставлять пилоты в поток данных",
            variable=self.wf_pil_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(pil_f, text="Период (символов данных):").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(pil_f, textvariable=self.wf_pil_period_var, width=10).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(pil_f, text="напр. 256", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)
        ttk.Label(pil_f, text="Длина пилота (символов):").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(pil_f, textvariable=self.wf_pil_len_var, width=10).grid(row=2, column=1, padx=8, sticky=tk.W)
        ttk.Label(pil_f, text="напр. 16", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=2, column=2, sticky=tk.W)
        ttk.Label(pil_f, text="Seed:").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Entry(pil_f, textvariable=self.wf_pil_seed_var, width=12).grid(row=3, column=1, padx=8, sticky=tk.W)

        plot_f = ttk.LabelFrame(parent, text="📈 Метрики / графики (waveform)", padding=10)
        plot_f.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Checkbutton(
            plot_f, text="✓ Созвездие до/после EQ (для последней точки SNR)",
            variable=self.wf_plot_const_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(plot_f, text="Точек для scatter:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(plot_f, textvariable=self.wf_const_points_var, width=10).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(plot_f, text="100..20000", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)
        ttk.Label(plot_f, text="EVM chunk (symbols):").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(plot_f, textvariable=self.wf_evm_chunk_var, width=10).grid(row=2, column=1, padx=8, sticky=tk.W)
        ttk.Label(plot_f, text="напр. 256", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=2, column=2, sticky=tk.W)

        timing_f = ttk.LabelFrame(parent, text="⏱ Timing recovery (Gardner)", padding=10)
        timing_f.pack(fill=tk.X, padx=10, pady=(0, 12))
        ttk.Checkbutton(
            timing_f, text="✓ Включить timing recovery (Gardner TED)",
            variable=self.wf_timing_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(timing_f, text="alpha:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(timing_f, textvariable=self.wf_timing_alpha_var, width=10).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(timing_f, text="напр. 0.01", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)
        ttk.Label(timing_f, text="beta:").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(timing_f, textvariable=self.wf_timing_beta_var, width=10).grid(row=2, column=1, padx=8, sticky=tk.W)
        ttk.Label(timing_f, text="напр. 1e-4", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=2, column=2, sticky=tk.W)
        ttk.Label(timing_f, text="init offset (samples):").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Entry(timing_f, textvariable=self.wf_timing_init_var, width=10).grid(row=3, column=1, padx=8, sticky=tk.W)
        ttk.Label(timing_f, text="0..sps (напр. 1.5)", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=3, column=2, sticky=tk.W)
        ttk.Checkbutton(
            timing_f, text="✓ Auto init offset (по eye opening)",
            variable=self.wf_timing_auto_var,
        ).grid(row=4, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))

        ti_f = ttk.LabelFrame(parent, text="🧨 Timing impairments (offset/drift/jitter на отсчётах)", padding=10)
        ti_f.pack(fill=tk.X, padx=10, pady=(0, 12))
        ttk.Checkbutton(
            ti_f, text="✓ Включить time-warp (ошибки дискретизации)",
            variable=self.wf_ti_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(ti_f, text="Offset (samples):").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(ti_f, textvariable=self.wf_ti_offset_var, width=10).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(ti_f, text="напр. 1.5", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)
        ttk.Label(ti_f, text="Drift (ppm):").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(ti_f, textvariable=self.wf_ti_drift_ppm_var, width=10).grid(row=2, column=1, padx=8, sticky=tk.W)
        ttk.Label(ti_f, text="напр. 50", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=2, column=2, sticky=tk.W)
        ttk.Label(ti_f, text="Jitter std (samples):").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Entry(ti_f, textvariable=self.wf_ti_jitter_var, width=10).grid(row=3, column=1, padx=8, sticky=tk.W)
        ttk.Label(ti_f, text="напр. 0.05", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=3, column=2, sticky=tk.W)
        ttk.Label(ti_f, text="Jitter mode:").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(
            ti_f, textvariable=self.wf_ti_mode_var,
            values=["white", "pll", "wiener"], state="readonly", width=10,
        ).grid(row=4, column=1, padx=8, sticky=tk.W)
        ttk.Label(ti_f, text="pll = коррелированный", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=4, column=2, sticky=tk.W)
        ttk.Label(ti_f, text="PLL bw norm:").grid(row=5, column=0, sticky=tk.W, pady=4)
        ttk.Entry(ti_f, textvariable=self.wf_ti_bw_var, width=10).grid(row=5, column=1, padx=8, sticky=tk.W)
        ttk.Label(ti_f, text="0..0.5 (напр. 0.01)", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=5, column=2, sticky=tk.W)
        ttk.Label(ti_f, text="Seed:").grid(row=6, column=0, sticky=tk.W, pady=4)
        ttk.Entry(ti_f, textvariable=self.wf_ti_seed_var, width=10).grid(row=6, column=1, padx=8, sticky=tk.W)

        eye_f = ttk.LabelFrame(parent, text="👁 Eye diagram", padding=10)
        eye_f.pack(fill=tk.X, padx=10, pady=(0, 12))
        ttk.Checkbutton(
            eye_f, text="✓ Сохранять eye diagram до/после timing recovery (последняя точка SNR)",
            variable=self.wf_eye_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(eye_f, text="Трасс:").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(eye_f, textvariable=self.wf_eye_traces_var, width=10).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(eye_f, text="20..2000", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)

        pn_f = ttk.LabelFrame(parent, text="🌀 Phase noise (Wiener на отсчётах)", padding=10)
        pn_f.pack(fill=tk.X, padx=10, pady=(0, 12))
        ttk.Checkbutton(
            pn_f, text="✓ Включить phase noise",
            variable=self.wf_pn_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(pn_f, text="Step std (deg/sample):").grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Entry(pn_f, textvariable=self.wf_pn_step_deg_var, width=10).grid(row=1, column=1, padx=8, sticky=tk.W)
        ttk.Label(pn_f, text="напр. 0.2", foreground=ACCENT_YELLOW, font=("Segoe UI", 8)).grid(row=1, column=2, sticky=tk.W)
        ttk.Label(pn_f, text="Seed:").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(pn_f, textvariable=self.wf_pn_seed_var, width=10).grid(row=2, column=1, padx=8, sticky=tk.W)

        adc_f = ttk.LabelFrame(parent, text="🎚 AGC + Clipping + ADC (quantization)", padding=10)
        adc_f.pack(fill=tk.X, padx=10, pady=(0, 12))
        ttk.Checkbutton(
            adc_f, text="✓ Включить ADC фронтенд (AGC/клиппинг/квантование)",
            variable=self.wf_adc_enabled_var,
        ).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))

        ttk.Checkbutton(adc_f, text="DC offset", variable=self.wf_dc_enabled_var).grid(row=1, column=0, sticky=tk.W, pady=4)
        ttk.Label(adc_f, text="dc_i:").grid(row=1, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_dc_i_var, width=10).grid(row=1, column=2, padx=8, sticky=tk.W)
        ttk.Label(adc_f, text="dc_q:").grid(row=2, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_dc_q_var, width=10).grid(row=2, column=2, padx=8, sticky=tk.W)

        ttk.Checkbutton(adc_f, text="IQ imbalance", variable=self.wf_iq_enabled_var).grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Label(adc_f, text="amp dB:").grid(row=3, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_iq_amp_db_var, width=10).grid(row=3, column=2, padx=8, sticky=tk.W)
        ttk.Label(adc_f, text="phase deg:").grid(row=4, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_iq_phase_deg_var, width=10).grid(row=4, column=2, padx=8, sticky=tk.W)

        ttk.Checkbutton(adc_f, text="AGC", variable=self.wf_agc_enabled_var).grid(row=5, column=0, sticky=tk.W, pady=4)
        ttk.Label(adc_f, text="target RMS:").grid(row=5, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_agc_target_var, width=10).grid(row=5, column=2, padx=8, sticky=tk.W)

        ttk.Label(adc_f, text="min gain:").grid(row=6, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_agc_min_gain_var, width=10).grid(row=6, column=2, padx=8, sticky=tk.W)
        ttk.Label(adc_f, text="max gain:").grid(row=7, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_agc_max_gain_var, width=10).grid(row=7, column=2, padx=8, sticky=tk.W)

        ttk.Checkbutton(adc_f, text="Clipping", variable=self.wf_clip_enabled_var).grid(row=8, column=0, sticky=tk.W, pady=4)
        ttk.Label(adc_f, text="clip level:").grid(row=8, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_clip_level_var, width=10).grid(row=8, column=2, padx=8, sticky=tk.W)

        ttk.Checkbutton(adc_f, text="Quantization", variable=self.wf_q_enabled_var).grid(row=9, column=0, sticky=tk.W, pady=4)
        ttk.Label(adc_f, text="bits:").grid(row=9, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_adc_bits_var, width=10).grid(row=9, column=2, padx=8, sticky=tk.W)
        ttk.Label(adc_f, text="full scale:").grid(row=10, column=1, sticky=tk.W, pady=4)
        ttk.Entry(adc_f, textvariable=self.wf_adc_fs_var, width=10).grid(row=10, column=2, padx=8, sticky=tk.W)

    def _open_recovery_panel(self) -> None:
        """
        Открывает окно сравнения оригинал vs восстановленный текст.
        Работает с последними результатами text-симуляции.
        """
        # Найти последний text-результат с полями text/original_text
        results = self.current_results
        if not results:
            messagebox.showwarning("Внимание", "Сначала запустите симуляцию в режиме «Текст»")
            return

        # Ищем последний SNR-результат содержащий текст
        text_result = None
        for r in reversed(results):
            if r.get("original_text") and r.get("text"):
                text_result = r
                break

        if text_result is None:
            messagebox.showwarning("Внимание", "Нет текстовых результатов. Запустите в режиме «Текст».")
            return

        original  = text_result["original_text"]
        recovered = text_result["text"]
        tc        = text_result.get("text_comparison", {})
        rs        = text_result.get("recovery_stats")

        # ── Окно ────────────────────────────────────────────────────────────
        win = tk.Toplevel(self.root)
        win.title("Восстановление текста — сравнение")
        win.geometry("900x620")
        win.configure(bg=DARK_BG)

        # Заголовок с метриками
        hdr = ttk.Frame(win)
        hdr.pack(fill=tk.X, padx=12, pady=(10, 4))

        cer     = tc.get("correct_percentage", 0)
        snr_val = text_result.get("snr", "?")
        ber_val = text_result.get("ber", 0)

        ttk.Label(
            hdr,
            text=f"Eb/N0 = {snr_val:.1f} дБ  |  BER = {ber_val:.2e}  |  "
                 f"CER = {cer:.1f}%  |  "
                 f"Символов: {tc.get('compared_chars', 0)} сравнено, "
                 f"{tc.get('correct_chars', 0)} верных",
            foreground=ACCENT_CYAN,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W)

        if rs:
            ttk.Label(
                hdr,
                text=(
                    f"Восстановление:  OK={rs['chars_ok']}  "
                    f"Исправлено={rs['chars_fixed']}  "
                    f"Потеряно={rs['chars_lost']}  "
                    f"Рейтинг={rs['recovery_rate']:.1%}  "
                    f"Время={rs['repair_ms']:.1f} мс"
                ),
                foreground=ACCENT_GREEN,
                font=("Segoe UI", 9),
            ).pack(anchor=tk.W, pady=(2, 0))

        ttk.Separator(win, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=6)

        # Два текстовых поля
        panes = ttk.Frame(win)
        panes.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))
        panes.columnconfigure(0, weight=1)
        panes.columnconfigure(1, weight=1)

        for col, (title, content, color) in enumerate([
            ("📄 ОРИГИНАЛ",      original,  ACCENT_CYAN),
            ("🔧 ВОССТАНОВЛЕННЫЙ", recovered, ACCENT_GREEN),
        ]):
            lbl_f = ttk.Frame(panes)
            lbl_f.grid(row=0, column=col, sticky="nsew", padx=(0 if col else 0, 6 if col == 0 else 0))
            panes.rowconfigure(0, weight=1)

            ttk.Label(lbl_f, text=title, foreground=color,
                      font=("Segoe UI", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))

            txt_widget = scrolledtext.ScrolledText(
                lbl_f,
                bg="#0d1b2a", fg=TEXT_COLOR,
                font=("Consolas", 9),
                wrap=tk.WORD,
                height=24,
            )
            txt_widget.pack(fill=tk.BOTH, expand=True)
            txt_widget.insert(tk.END, content)

            # Подсветка повреждённых символов (только для восстановленного)
            if col == 1:
                txt_widget.tag_configure("damaged", foreground=ACCENT_PINK,
                                          background="#2a0010")
                txt_widget.tag_configure("fixed",   foreground=ACCENT_YELLOW,
                                          background="#1a1a00")
                self._highlight_damaged(txt_widget, original, recovered)

            txt_widget.config(state=tk.DISABLED)

        # Легенда
        leg_f = ttk.Frame(win)
        leg_f.pack(fill=tk.X, padx=12, pady=(0, 8))
        for color, bg, text in [
            (ACCENT_PINK,   "#2a0010", "■ Повреждён (не совпадает с оригиналом)"),
            (ACCENT_YELLOW, "#1a1a00", "■ Восстановлен (был U+FFFD, теперь читаем)"),
            (TEXT_COLOR,    DARK_BG,   "■ Верный символ"),
        ]:
            tk.Label(leg_f, text=text, bg=bg, fg=color,
                     font=("Segoe UI", 8), padx=6).pack(side=tk.LEFT, padx=4)

    @staticmethod
    def _highlight_damaged(
        widget: scrolledtext.ScrolledText,
        original: str,
        recovered: str,
    ) -> None:
        """
        Подсвечивает в виджете символы восстановленного текста:
          - розовый (#damaged): символ отличается от оригинала
          - жёлтый  (#fixed):   символ был «?» но стал читаемым
        """
        n = min(len(original), len(recovered))
        for i in range(n):
            o_ch = original[i]
            r_ch = recovered[i]
            if r_ch == "?":
                tag = "damaged"
            elif o_ch != r_ch:
                tag = "damaged"
            else:
                continue
            # Позиция в Text: строка 1, символ i
            # scrolledtext считает с "1.0"
            try:
                idx_start = f"1.0 + {i} chars"
                idx_end   = f"1.0 + {i + 1} chars"
                widget.tag_add(tag, idx_start, idx_end)
            except Exception:
                pass

    def _build_plot_tab(self, parent: ttk.Frame) -> None:
        ttk.Checkbutton(parent, text="✓ Теоретические кривые AWGN",
                        variable=self.show_theo_var).pack(anchor=tk.W, padx=15, pady=10)
        ttk.Checkbutton(parent, text="✓ Теоретические кривые Rayleigh",
                        variable=self.show_rayleigh_theo_var).pack(anchor=tk.W, padx=15, pady=(0, 10))
        ttk.Label(parent,
                  text=("AWGN-кривые: точные формулы Proakis\n"
                        "для M-PSK / M-QAM в аддитивном шуме.\n\n"
                        "Rayleigh-кривые: теоретический BER\n"
                        "для плоских замираний (MGF-подход).\n"
                        "Позволяет сравнить симуляцию с теорией\n"
                        "и оценить потери от замираний."),
                  justify=tk.LEFT, foreground=ACCENT_YELLOW).pack(anchor=tk.W, padx=15, pady=10)

    def _build_metrics_table(self, parent: ttk.Frame) -> None:
        cols = ("snr", "ber", "ber_post", "ser", "theo_ber", "epf", "cer")
        self.tree = ttk.Treeview(parent, columns=cols, show="headings", height=5)
        for col, heading, width in [
            ("snr",      "Eb/N0\n(дБ)",    80),
            ("ber",      "BER\n(пост-дш)", 100),
            ("ber_post", "BER\n(пре-дш)",  100),
            ("ser",      "SER\n(эксп.)",   100),
            ("theo_ber", "BER\n(теор.)",   110),
            ("epf",      "Распр.\nошибок",  90),
            ("cer",      "CER\n(%)",         90),
        ]:
            self.tree.heading(col, text=heading)
            self.tree.column(col, width=width, anchor=tk.CENTER)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    # ── Вспомогательные UI ────────────────────────────────────────────────────

    def browse_file(self) -> None:
        fn = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if fn:
            self.text_file_var.set(fn)

    # ── Конфигурация ──────────────────────────────────────────────────────────

    def build_config(self) -> dict | None:
        """
        Собирает конфигурацию из виджетов.
        При ошибке показывает список всех проблемных полей и возвращает None.
        """
        errors: list[str] = []

        def _float(var: tk.StringVar, label: str, **cond: object) -> float | None:
            try:
                val = float(var.get())
                for check, msg in cond.items():
                    if check == "gt" and val <= cond["gt"]:
                        errors.append(f"{label}: должно быть > {cond['gt']}")
                return val
            except ValueError:
                errors.append(f"{label}: не является числом ({var.get()!r})")
                return None

        def _int(var: tk.StringVar, label: str, min_val: int = 1) -> int | None:
            try:
                val = int(var.get())
                if val < min_val:
                    errors.append(f"{label}: должно быть ≥ {min_val}")
                return val
            except ValueError:
                errors.append(f"{label}: не является целым ({var.get()!r})")
                return None

        start = _float(self.ebn0_start_var, "Eb/N0 От")
        stop  = _float(self.ebn0_stop_var,  "Eb/N0 До")
        step  = _float(self.ebn0_step_var,  "Шаг Eb/N0")

        if start is not None and stop is not None and start >= stop:
            errors.append("Eb/N0: 'От' должно быть меньше 'До'")
        if step is not None and step <= 0:
            errors.append("Шаг Eb/N0: должен быть > 0")

        num_bits  = _int(self.num_bits_var,        "Количество бит")
        num_sims  = _int(self.num_simulations_var,  "Число усреднений")
        max_len   = _int(self.max_text_length_var,  "Макс. длина")
        num_reps  = _int(self.num_repetitions_var,  "Повторений")

        if errors:
            messagebox.showerror("Ошибка ввода", "\n".join(errors))
            return None

        rng = np.arange(start, stop + step / 2, step).tolist()

        coding_type_val = self.coding_type_var.get()
        if coding_type_val == "hamming":
            n, k = 7, 4
        elif coding_type_val == "reed-solomon":
            n, k = 0, 0  # не используются для RS
        else:  # turbo — n/k не используются, скорость фиксирована ≈ 1/3
            n, k = 0, 0
        try:
            max_adaptive = int(self.max_adaptive_bits_var.get())
        except ValueError:
            max_adaptive = 1_000_000
        try:
            early_stop = float(self.early_stop_ber_var.get())
        except ValueError:
            early_stop = 0.0
        try:
            per_packet = int(self.per_packet_var.get())
        except ValueError:
            per_packet = 1024
        try:
            turbo_iter = int(self.turbo_iterations_var.get())
        except ValueError:
            turbo_iter = 6
        try:
            turbo_block = int(self.turbo_block_size_var.get())
        except ValueError:
            turbo_block = 128

        # Waveform параметры (ошибки не фатальны — используем safe defaults)
        try:
            wf_sps = int(self.wf_sps_var.get())
            if wf_sps < 2:
                wf_sps = 8
        except ValueError:
            wf_sps = 8
        try:
            wf_beta = float(self.wf_rrc_beta_var.get())
            if wf_beta < 0.0 or wf_beta > 1.0:
                wf_beta = 0.35
        except ValueError:
            wf_beta = 0.35
        try:
            wf_span = int(self.wf_rrc_span_var.get())
            if wf_span < 2:
                wf_span = 10
        except ValueError:
            wf_span = 10
        try:
            wf_cfo_norm = float(self.wf_cfo_norm_var.get())
        except ValueError:
            wf_cfo_norm = 0.0

        try:
            wf_prelen = int(self.wf_cfo_prelen_var.get())
            if wf_prelen < 8:
                wf_prelen = 64
        except ValueError:
            wf_prelen = 64

        def _parse_int_list(s: str, default: list[int]) -> list[int]:
            if not s or not str(s).strip():
                return list(default)
            parts = [p.strip() for p in str(s).replace(";", ",").split(",") if p.strip()]
            out: list[int] = []
            for p in parts:
                try:
                    out.append(int(p))
                except ValueError:
                    return list(default)
            return out or list(default)

        def _parse_float_list(s: str, default: list[float]) -> list[float]:
            if not s or not str(s).strip():
                return list(default)
            parts = [p.strip() for p in str(s).replace(";", ",").split(",") if p.strip()]
            out: list[float] = []
            for p in parts:
                try:
                    out.append(float(p))
                except ValueError:
                    return list(default)
            return out or list(default)

        # TDL + EQ (safe defaults)
        tdl_delays = _parse_float_list(self.wf_tdl_delays_var.get(), [0.0, 1.0, 3.0])
        tdl_powers = _parse_float_list(self.wf_tdl_powers_var.get(), [0.0, -3.0, -6.0])
        try:
            tdl_seed_str = (self.wf_tdl_seed_var.get() or "").strip()
            tdl_seed = int(tdl_seed_str) if tdl_seed_str else None
        except ValueError:
            tdl_seed = None
        try:
            tdl_frac_taps = int(self.wf_tdl_frac_taps_var.get())
        except ValueError:
            tdl_frac_taps = 21
        if tdl_frac_taps < 5:
            tdl_frac_taps = 21
        if tdl_frac_taps % 2 == 0:
            tdl_frac_taps += 1

        eq_kind = (self.wf_eq_kind_var.get() or "mmse").lower()
        if eq_kind not in ("zf", "mmse"):
            eq_kind = "mmse"
        eq_est = (self.wf_eq_est_var.get() or "ideal").lower()
        if eq_est not in ("ideal", "training_ls", "pilot_ls", "preamble_ls"):
            eq_est = "ideal"
        try:
            eq_ls_reg = float(self.wf_eq_ls_reg_var.get())
        except ValueError:
            eq_ls_reg = 1e-3
        if eq_ls_reg < 0:
            eq_ls_reg = 1e-3
        try:
            eq_len = int(self.wf_eq_len_var.get())
        except ValueError:
            eq_len = 9
        eq_len = max(1, min(63, eq_len))
        try:
            eq_delay_str = (self.wf_eq_delay_var.get() or "").strip()
            eq_delay = int(eq_delay_str) if eq_delay_str else None
        except ValueError:
            eq_delay = None
        try:
            eq_chan_taps = int(self.wf_eq_chan_taps_var.get())
        except ValueError:
            eq_chan_taps = 32
        eq_chan_taps = max(16, min(128, eq_chan_taps))

        # training + pilots (safe defaults)
        try:
            tr_len = int(self.wf_tr_len_var.get())
        except ValueError:
            tr_len = 128
        tr_len = max(16, min(4096, tr_len))
        try:
            tr_seed = int((self.wf_tr_seed_var.get() or "12345").strip())
        except ValueError:
            tr_seed = 12345
        try:
            pil_period = int(self.wf_pil_period_var.get())
        except ValueError:
            pil_period = 256
        pil_period = max(16, min(1000000, pil_period))
        try:
            pil_len = int(self.wf_pil_len_var.get())
        except ValueError:
            pil_len = 16
        pil_len = max(4, min(2048, pil_len))
        try:
            pil_seed = int((self.wf_pil_seed_var.get() or "4242").strip())
        except ValueError:
            pil_seed = 4242

        try:
            const_pts = int(self.wf_const_points_var.get())
        except ValueError:
            const_pts = 2000
        const_pts = max(100, min(20000, const_pts))
        try:
            evm_chunk = int(self.wf_evm_chunk_var.get())
        except ValueError:
            evm_chunk = 256
        evm_chunk = max(32, min(8192, evm_chunk))

        try:
            tr_alpha = float(self.wf_timing_alpha_var.get())
        except ValueError:
            tr_alpha = 0.01
        try:
            tr_beta = float(self.wf_timing_beta_var.get())
        except ValueError:
            tr_beta = 0.0001
        try:
            tr_init = float(self.wf_timing_init_var.get())
        except ValueError:
            tr_init = 0.0

        # timing impairments + eye diagram
        try:
            ti_offset = float(self.wf_ti_offset_var.get())
        except ValueError:
            ti_offset = 0.0
        try:
            ti_drift_ppm = float(self.wf_ti_drift_ppm_var.get())
        except ValueError:
            ti_drift_ppm = 0.0
        try:
            ti_jitter = float(self.wf_ti_jitter_var.get())
        except ValueError:
            ti_jitter = 0.0
        try:
            ti_seed = int((self.wf_ti_seed_var.get() or "").strip())
        except ValueError:
            ti_seed = 2026
        ti_mode = (self.wf_ti_mode_var.get() or "white").lower()
        if ti_mode not in ("white", "pll", "wiener"):
            ti_mode = "white"
        try:
            ti_bw = float(self.wf_ti_bw_var.get())
        except ValueError:
            ti_bw = 0.01
        if ti_bw < 0:
            ti_bw = 0.01
        try:
            eye_traces = int(self.wf_eye_traces_var.get())
        except ValueError:
            eye_traces = 200
        eye_traces = max(20, min(2000, eye_traces))

        # phase noise (samples) safe defaults
        try:
            pn_step_deg = float(self.wf_pn_step_deg_var.get())
        except ValueError:
            pn_step_deg = 0.2
        try:
            pn_seed = int((self.wf_pn_seed_var.get() or "").strip())
        except ValueError:
            pn_seed = 777

        # ADC/AGC/clip/quant safe defaults
        def _flt(s: str, default: float) -> float:
            try:
                return float(s)
            except ValueError:
                return float(default)
        def _intv(s: str, default: int) -> int:
            try:
                return int(s)
            except ValueError:
                return int(default)

        agc_target = _flt(self.wf_agc_target_var.get(), 1.0)
        agc_min_g  = _flt(self.wf_agc_min_gain_var.get(), 0.01)
        agc_max_g  = _flt(self.wf_agc_max_gain_var.get(), 100.0)
        clip_level = _flt(self.wf_clip_level_var.get(), 1.2)
        adc_bits   = _intv(self.wf_adc_bits_var.get(), 10)
        adc_fs     = _flt(self.wf_adc_fs_var.get(), 1.0)
        dc_i       = _flt(self.wf_dc_i_var.get(), 0.0)
        dc_q       = _flt(self.wf_dc_q_var.get(), 0.0)
        iq_amp_db  = _flt(self.wf_iq_amp_db_var.get(), 0.0)
        iq_phi_deg = _flt(self.wf_iq_phase_deg_var.get(), 0.0)

        pll_mode = (self.wf_pll_mode_var.get() or "medium").lower()
        if pll_mode == "slow":
            pll_alpha, pll_beta = 0.01, 0.0001
        elif pll_mode == "fast":
            pll_alpha, pll_beta = 0.05, 0.0012
        else:
            pll_alpha, pll_beta = 0.02, 0.0004

        cv = self.channel_vars
        channel_config = {
            "awgn":             {"enabled": True},
            "rayleigh": {
                "enabled":            cv["rayleigh_enabled"].get(),
                "n_rays":             int(cv["rayleigh_rays"].get()),
                "normalized_doppler": float(cv["rayleigh_doppler"].get()),
            },
            "phase_noise": {
                "enabled":              cv["phase_enabled"].get(),
                "phase_noise_std_deg":  float(cv["phase_std_deg"].get()),
            },
            "impulse_noise": {
                "enabled":             cv["impulse_noise_enabled"].get(),
                "mode":                      cv["impulse_noise_mode"].get(),
                "impulse_snr_dB":            float(cv["impulse_noise_snr_db"].get()),
                "impulse_probability":       float(cv["impulse_noise_prob"].get()),
                "burst_start_probability":   float(cv["impulse_noise_burst_p"].get()),
                "burst_len_min":             int(cv["impulse_noise_burst_len_min"].get()),
                "burst_len_max":             int(cv["impulse_noise_burst_len_max"].get()),
                "burst_gap_min":             int(cv["impulse_noise_gap_min"].get()),
                "burst_gap_max":             int(cv["impulse_noise_gap_max"].get()),
                "in_burst_pulse_probability": float(cv["impulse_noise_inburst_p"].get()),
                "pulse_width_symbols":       int(cv["impulse_noise_width"].get()),
                "radar_burst_period_symbols":  int(cv["impulse_noise_radar_period"].get()),
                "radar_pulses_per_burst":      int(cv["impulse_noise_radar_pulses"].get()),
                "radar_pulse_spacing_symbols": int(cv["impulse_noise_radar_spacing"].get()),
                "radar_pulse_width_symbols":   int(cv["impulse_noise_radar_width"].get()),
                "radar_pulse_jitter_symbols":  int(cv["impulse_noise_radar_jitter"].get()),
            },
        }

        # Параметры Reed-Solomon
        rs_nsym  = int(self.rs_nsym_var.get())
        rs_nsize = int(self.rs_nsize_var.get())
        rs_fcr   = int(self.rs_fcr_var.get())
        rs_prim  = int(self.rs_prim_var.get(), 16)   # шестнадцатеричный ввод

        return {
            "profile": self.profile_var.get(),
            "simulation_mode": self.mode_var.get(),
            "modulation": {
                "type":          self.modulation_type_var.get(),
                "order":         int(self.modulation_order_var.get()),
                "use_gray_code": self.gray_code_var.get(),
            },
            "coding": {
                "enabled":           self.coding_enabled_var.get(),
                "type":              coding_type_val,
                "n":                 n,
                "k":                 k,
                "turbo_iterations":  turbo_iter,
                "turbo_block_size":  turbo_block,
                "rs_nsym":           rs_nsym,
                "rs_nsize":          rs_nsize,
                "rs_fcr":            rs_fcr,
                "rs_prim":           rs_prim,
            },
            "channel": channel_config,
            "text_settings": {
                "text_file":       self.text_file_var.get(),
                "text_encoding":   self.text_encoding_var.get(),
                "max_text_length": max_len,
                "num_repetitions": num_reps,
            },
            "random_settings": {
                "num_bits":           num_bits,
                "num_simulations":    num_sims,
                "max_adaptive_bits":  max_adaptive,
            },
            "ebn0_dB_range":    rng,
            "early_stop_ber":   early_stop,
            "per_settings": {
                "enabled":     self.per_enabled_var.get(),
                "packet_size": per_packet,
            },
            "encryption": {
                "enabled":  self.enc_enabled_var.get(),
                "type":     self.enc_type_var.get(),
                "aes_mode": self.enc_aes_mode_var.get(),
                "key_hex":  self.enc_key_var.get().strip() or None,
            },
            "show_theo":          self.show_theo_var.get(),
            "show_rayleigh_theo": self.show_rayleigh_theo_var.get(),
            "interleaving": {
                "enabled": self.il_enabled_var.get(),
                "depth":   int(self.il_depth_var.get()),
            },
            "waveform": {
                "enabled":          self.wf_enabled_var.get(),
                "sps":              wf_sps,
                "rrc_beta":         wf_beta,
                "rrc_span_symbols": wf_span,
                "cfo": {
                    "enabled":  self.wf_cfo_enabled_var.get(),
                    "cfo_norm": wf_cfo_norm,
                    "recovery_enabled": self.wf_cfo_rec_var.get(),
                    "pll_alpha": pll_alpha,
                    "pll_beta":  pll_beta,
                    "preamble_enabled": self.wf_cfo_preamble_var.get(),
                    "preamble_half_len_symbols": wf_prelen,
                    "loop_type": (self.wf_cfo_loop_var.get() or "dd_pll"),
                    "estimation_source": (self.wf_cfo_src_var.get() or "preamble"),
                },
                "tdl": {
                    "enabled": self.wf_tdl_enabled_var.get(),
                    "delays_symbols": tdl_delays,
                    "powers_dB": tdl_powers,
                    "fading": "rayleigh",
                    "seed": tdl_seed,
                    "frac_taps": tdl_frac_taps,
                },
                "equalizer": {
                    "enabled": self.wf_eq_enabled_var.get(),
                    "kind": eq_kind,
                    "eq_len": eq_len,
                    "delay": eq_delay,
                    "channel_taps": eq_chan_taps,
                    "estimation": eq_est,
                    "ls_reg": eq_ls_reg,
                },
                "training": {
                    "enabled": self.wf_tr_enabled_var.get(),
                    "length_symbols": tr_len,
                    "seed": tr_seed,
                },
                "pilots": {
                    "enabled": self.wf_pil_enabled_var.get(),
                    "period_symbols": pil_period,
                    "length_symbols": pil_len,
                    "seed": pil_seed,
                },
                "plots": {
                    "constellation": self.wf_plot_const_var.get(),
                    "constellation_points": const_pts,
                    "eye_diagram": self.wf_eye_enabled_var.get(),
                    "eye_traces": eye_traces,
                    "evm_chunk_symbols": evm_chunk,
                },
                "timing": {
                    "enabled": self.wf_timing_enabled_var.get(),
                    "method": "gardner",
                    "alpha": tr_alpha,
                    "beta": tr_beta,
                    "init_offset_samples": tr_init,
                    "auto_init": self.wf_timing_auto_var.get(),
                },
                "timing_impairments": {
                    "enabled": self.wf_ti_enabled_var.get(),
                    "offset_samples": ti_offset,
                    "drift_ppm": ti_drift_ppm,
                    "jitter_std_samples": ti_jitter,
                    "seed": ti_seed,
                    "jitter_mode": ti_mode,
                    "jitter_pll_bw_norm": ti_bw,
                },
                "phase_noise": {
                    "enabled": self.wf_pn_enabled_var.get(),
                    "step_std_deg": pn_step_deg,
                    "seed": pn_seed,
                },
                "adc": {
                    "enabled": self.wf_adc_enabled_var.get(),
                    "dc_enabled": self.wf_dc_enabled_var.get(),
                    "dc_i": dc_i,
                    "dc_q": dc_q,
                    "iq_imbalance_enabled": self.wf_iq_enabled_var.get(),
                    "iq_amp_imbalance_db": iq_amp_db,
                    "iq_phase_imbalance_deg": iq_phi_deg,
                    "agc_enabled": self.wf_agc_enabled_var.get(),
                    "agc_target_rms": agc_target,
                    "agc_min_gain": agc_min_g,
                    "agc_max_gain": agc_max_g,
                    "clipping_enabled": self.wf_clip_enabled_var.get(),
                    "clip_level": clip_level,
                    "quantization_enabled": self.wf_q_enabled_var.get(),
                    "n_bits": adc_bits,
                    "full_scale": adc_fs,
                },
            },

        }

    # ── Управление симуляцией ─────────────────────────────────────────────────

    def start_simulation(self) -> None:
        for r in self.tree.get_children():
            self.tree.delete(r)
        self.clear_plot()

        config = self.build_config()
        if config is None:
            return

        self._log_filename = self._setup_logging()
        self._set_running_state(True)

        self.sim_thread = threading.Thread(
            target=self.run_simulation,
            args=(config, config["simulation_mode"]),
            daemon=True,
        )
        self.sim_thread.start()

    def run_simulation(self, config: dict, mode: str) -> None:
        start_time = time.time()
        rng        = config["ebn0_dB_range"]
        results: list[dict] = []
        total   = len(rng)

        logging.info("=" * 60)
        logging.info("НАЧАЛО СИМУЛЯЦИИ")
        logging.info(f"Режим: {mode} | Модуляция: "
                     f"{config['modulation']['type']}-{config['modulation']['order']}")

        # Для text-режима загружаем файл один раз
        text: str | None = None
        if mode == "text":
            try:
                ts = config["text_settings"]
                with open(ts["text_file"], encoding=ts["text_encoding"]) as f:
                    text = f.read()
                logging.info(f"Текст загружен: {ts['text_file']} ({len(text)} симв.)")
            except OSError as e:
                self.queue.put(("error", f"Ошибка чтения файла: {e}"))
                self.queue.put(("done",))
                return

        for idx, snr in enumerate(rng):
            if not self.running:
                logging.warning("Симуляция прервана пользователем")
                break

            # Прогресс и ETA
            elapsed = time.time() - start_time
            pct = idx / total * 100
            eta_str = ""
            if idx > 0:
                eta_sec = elapsed / idx * (total - idx)
                eta_str = f"ETA: {int(eta_sec)}s" if eta_sec < 3600 else f"ETA: {eta_sec/60:.1f}m"
            else:
                eta_str = "Расчёт ETA..."
            self.queue.put(("progress", pct, f"SNR={snr:.1f} дБ ({idx+1}/{total})", eta_str))

            log_once = (idx == 0)   # конфиг логируется только на первой итерации

            if mode == "random":
                num_sim = config["random_settings"]["num_simulations"]
                prev_ber = results[-1]["ber"] if results else None
                batch = [
                    simulate_transmission(config, snr, log_config_once=log_once,
                                          prev_ber=prev_ber)
                    for i in range(num_sim)
                    if self.running
                ]
                if not batch:
                    break
                avg = _average_stats(batch, snr)
                # Ранняя остановка
                if avg.get("early_stop", False):
                    results.append(avg)
                    self.queue.put(("add_row", avg))
                    logging.info(f"Ранняя остановка: BER={avg['ber']:.2e} < порога")
                    break

            else:  # text
                num_rep = config["text_settings"]["num_repetitions"]
                batch = [
                    simulate_text_transmission(config, text, snr, log_config_once=log_once)
                    for _ in range(num_rep)
                    if self.running
                ]
                if not batch:
                    break
                avg = _average_stats(batch, snr)
                # CER берём из text_comparison
                avg["cer"] = float(np.mean([
                    r["text_comparison"]["correct_percentage"] for r in batch
                ]))

            if not self.running:
                break

            results.append(avg)
            self.queue.put(("add_row", avg))

        if self.running and results:
            self.current_config  = config
            self.current_results = results
            elapsed = time.time() - start_time
            results_file = save_results_to_text(config, results, mode, execution_time=elapsed)
            # Авто-экспорт CSV рядом с текстовым отчётом
            csv_file = results_file.replace(".txt", ".csv")
            try:
                _auto_export_csv(results, csv_file)
                logging.info(f"CSV экспортирован: {csv_file}")
            except Exception as _csv_err:
                logging.warning(f"Не удалось сохранить CSV: {_csv_err}")
                csv_file = None
            self.queue.put(("plot_data", (config, results, mode)))
            self.queue.put(("progress", 100, f"Завершено за {elapsed:.1f}с", ""))
            info_msg = f"Результаты:\n{results_file}"
            if csv_file:
                info_msg += f"\nCSV: {csv_file}"
            self.queue.put(("info", info_msg))
            logging.info("=" * 60)
            logging.info(f"СИМУЛЯЦИЯ ЗАВЕРШЕНА за {elapsed:.1f} с")
            logging.info(f"Результаты сохранены в {results_file}")
        else:
            self.queue.put(("progress", 0, "Остановлено", ""))

        self.queue.put(("done",))

    def stop_simulation(self) -> None:
        self.running = False
        self.progress_label_var.set("Остановка...")

    # ── Очередь ───────────────────────────────────────────────────────────────

    def process_queue(self) -> None:
        try:
            while True:
                msg = self.queue.get_nowait()

                if msg[0] == "progress":
                    self.progress_var.set(msg[1])
                    self.progress_label_var.set(msg[2])
                    self.eta_var.set(msg[3] if len(msg) > 3 else "")

                elif msg[0] == "add_row":
                    s = msg[1]
                    cer = f"{s.get('cer', 0):.2f}" if s.get("cer", 0) else "—"
                    epf = s.get("error_propagation_factor", 1.0)
                    epf_str = f"{epf:.2f}" if s.get("encryption_enabled") else "—"
                    ber_post = s.get("ber_post_decrypt", s.get("ber", 0))
                    self.tree.insert("", tk.END, values=(
                        f"{s['snr']:.1f}",
                        f"{s['ber']:.2e}",
                        f"{ber_post:.2e}",
                        f"{s['ser']:.2e}",
                        f"{s.get('theoretical_ber', 0):.2e}",
                        epf_str,
                        cer,
                    ))
                    self.tree.see(self.tree.get_children()[-1])

                elif msg[0] == "plot_data":
                    config, results, mode = msg[1]
                    try:
                        _, fig = plot_and_save_results(
                            config, results, mode,
                            show_theoretical=self.show_theo_var.get(),
                            show_rayleigh_theo=self.show_rayleigh_theo_var.get(),
                        )
                        if fig is not None:
                            self.display_plot(fig)
                    except Exception as e:
                        logging.error(f"Ошибка построения графика: {e}")

                elif msg[0] == "info":
                    messagebox.showinfo("✓ Информация", msg[1])

                elif msg[0] == "error":
                    messagebox.showerror("✗ Ошибка", msg[1])

                elif msg[0] == "done":
                    self._set_running_state(False)

        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Queue error: {e}")
        finally:
            self.root.after(100, self.process_queue)

    # ── График ────────────────────────────────────────────────────────────────

    def display_plot(self, fig: object) -> None:
        self.clear_plot()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self._current_canvas = canvas
        self._current_fig    = fig

    def clear_plot(self) -> None:
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self._current_canvas = None
        self._current_fig    = None

    # ── Результаты ────────────────────────────────────────────────────────────

    def clear_results(self) -> None:
        for r in self.tree.get_children():
            self.tree.delete(r)
        self.clear_plot()
        self.progress_var.set(0)
        self.progress_label_var.set("Готов")
        self.eta_var.set("")
        self._set_running_state(False)
        self.current_results = None
        self.current_config  = None

    def save_results(self) -> None:
        if self.current_results is None:
            messagebox.showwarning("Внимание", "Нет результатов для сохранения")
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Сохранить результаты")
        dlg.geometry("350x150")
        dlg.configure(bg=DARK_BG)

        frm = ttk.Frame(dlg)
        frm.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        ttk.Label(frm, text="Имя конфигурации:", style="Title.TLabel").pack(anchor=tk.W, pady=(0, 5))
        name_var = tk.StringVar()
        entry = ttk.Entry(frm, textvariable=name_var, width=40)
        entry.pack(fill=tk.X, pady=(0, 15))
        entry.focus()

        def _save() -> None:
            name = name_var.get().strip() or None
            result_id = results_manager.save_results(
                self.current_config, self.current_results,
                self.mode_var.get(), name,
            )
            messagebox.showinfo("✓ Сохранено", f"ID: {result_id}")
            dlg.destroy()

        bf = ttk.Frame(frm)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="✓ Сохранить", command=_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="✗ Отмена", command=dlg.destroy).pack(side=tk.LEFT, padx=5)

    def show_history(self) -> None:
        hw = tk.Toplevel(self.root)
        hw.title("История симуляций")
        hw.geometry("700x500")
        hw.configure(bg=DARK_BG)

        frm = ttk.Frame(hw)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        ttk.Label(frm, text="📚 Сохранённые результаты", style="Title.TLabel",
                  foreground=ACCENT_CYAN).pack(anchor=tk.W, pady=(0, 10))

        records = results_manager.get_results_list()
        if not records:
            ttk.Label(frm, text="История пуста", foreground=ACCENT_YELLOW).pack(pady=20)
            return

        tf = ttk.Frame(frm)
        tf.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cols = ("name", "modulation", "coding", "mode", "points", "snr_range")
        tree = ttk.Treeview(tf, columns=cols, show="headings", height=10)
        for col, heading, width in [
            ("name",      "Имя",        150),
            ("modulation","Модуляция",  100),
            ("coding",    "Кодирование",100),
            ("mode",      "Режим",       60),
            ("points",    "Точек",        60),
            ("snr_range", "SNR (дБ)",   120),
        ]:
            tree.heading(col, text=heading)
            tree.column(col, width=width, anchor=tk.CENTER)

        for info in records:
            snr_str = f"{info['snr_range'][0]:.0f}-{info['snr_range'][1]:.0f}"
            tree.insert("", tk.END, iid=info["id"], values=(
                info["name"], info["modulation"], info["coding"],
                info["mode"], info["num_points"], snr_str,
            ))

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        def _load() -> None:
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Выбор", "Выберите результат")
                return
            try:
                config, results = results_manager.load_results(sel[0])
            except (KeyError, FileNotFoundError) as e:
                messagebox.showerror("Ошибка", str(e))
                return

            self.current_config  = config
            self.current_results = results

            for r in self.tree.get_children():
                self.tree.delete(r)
            self.clear_plot()

            for r in results:
                cer = f"{r.get('cer', 0):.2f}" if r.get("cer", 0) else "—"
                epf = r.get("error_propagation_factor", 1.0)
                epf_str = f"{epf:.2f}" if r.get("encryption_enabled") else "—"
                ber_post = r.get("ber_post_decrypt", r.get("ber", 0))
                self.tree.insert("", tk.END, values=(
                    f"{r['snr']:.1f}",
                    f"{r['ber']:.2e}",
                    f"{ber_post:.2e}",
                    f"{r.get('ser', 0):.2e}",
                    f"{r.get('theoretical_ber', 0):.2e}",
                    epf_str,
                    cer,
                ))

            try:
                _, fig = plot_and_save_results(
                    config, results, config["simulation_mode"],
                    show_theoretical=self.show_theo_var.get(),
                    show_rayleigh_theo=self.show_rayleigh_theo_var.get(),
                )
                if fig:
                    self.display_plot(fig)
            except Exception as e:
                logging.error(f"Ошибка графика при загрузке: {e}")

            messagebox.showinfo("✓", f"Загружены: {results_manager.index[sel[0]]['name']}")
            hw.destroy()

        def _delete() -> None:
            sel = tree.selection()
            if not sel:
                messagebox.showwarning("Выбор", "Выберите результат")
                return
            if messagebox.askyesno("Подтверждение", "Удалить результаты?"):
                results_manager.delete_results(sel[0])
                tree.delete(sel[0])

        def _compare() -> None:
            sel = tree.selection()
            if len(sel) < 2:
                messagebox.showwarning("Выбор", "Выберите минимум 2 результата")
                return
            try:
                comparison = results_manager.compare_results(list(sel))
            except Exception as e:
                messagebox.showerror("Ошибка", str(e))
                return
            cw = tk.Toplevel(hw)
            cw.title("Сравнение результатов")
            cw.geometry("800x500")
            cw.configure(bg=DARK_BG)
            txt = scrolledtext.ScrolledText(cw, bg="#1a1a2e", fg=TEXT_COLOR, font=("Courier", 9))
            txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            out = "=" * 100 + "\nСРАВНЕНИЕ РЕЗУЛЬТАТОВ\n" + "=" * 100 + "\n\n"
            for res in comparison["results"]:
                out += f"▶ {res['name']}\n"
                out += f"  Модуляция: {res['config']['modulation']} | Кодирование: {res['config']['coding']}\n"
                out += f"  {'SNR(дБ)':<10} {'BER':<12} {'SER':<12}\n"
                out += "-" * 50 + "\n"
                for d in res["data"]:
                    out += f"  {d['snr']:<10.1f} {d['ber']:<12.2e} {d['ser']:<12.2e}\n"
                out += "\n"
            txt.insert(tk.END, out)
            txt.config(state=tk.DISABLED)

        bf = ttk.Frame(frm)
        bf.pack(fill=tk.X)
        ttk.Button(bf, text="📂 Загрузить", command=_load).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="🗑 Удалить",   command=_delete).pack(side=tk.LEFT, padx=5)
        ttk.Button(bf, text="📊 Сравнить",  command=_compare).pack(side=tk.LEFT, padx=5)

    def export_results(self) -> None:
        if self.current_results is None:
            messagebox.showwarning("Внимание", "Нет результатов для экспорта")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("JSON файлы", "*.json"), ("PNG графики", "*.png")],
        )
        if not filename:
            return

        try:
            if filename.endswith(".csv"):
                with open(filename, "w", newline="", encoding="utf-8") as f:
                    if self.current_results:
                        flat = [_flatten_result(r) for r in self.current_results]
                        writer = csv.DictWriter(f, fieldnames=flat[0].keys())
                        writer.writeheader()
                        writer.writerows(flat)

            elif filename.endswith(".json"):
                with open(filename, "w", encoding="utf-8") as f:
                    json.dump(_to_python({
                        "config":  self.current_config,
                        "results": self.current_results,
                    }), f, indent=2, ensure_ascii=False)

            elif filename.endswith(".png"):
                if self._current_fig is not None:
                    self._current_fig.savefig(filename, dpi=150, facecolor="#1a1a2e")
                else:
                    messagebox.showwarning("Внимание", "Нет графика для сохранения")
                    return

            messagebox.showinfo("✓ Экспортировано", f"Файл: {filename}")
        except Exception as e:
            messagebox.showerror("✗ Ошибка", f"Ошибка экспорта: {e}")

    def export_csv(self) -> None:
        """
        Экспорт текущего прогона в CSV с метаданными.
        Использует results_manager.export_to_csv() для полного формата по ТЗ.
        Если текущий прогон не сохранён — предлагает сохранить сначала.
        """
        if self.current_results is None:
            messagebox.showwarning("Внимание", "Нет результатов для экспорта")
            return

        # Пробуем найти последний сохранённый результат
        records = results_manager.get_results_list()
        current_id: str | None = None
        if records:
            # Берём самый свежий — пользователь только что мог сохранить
            current_id = records[0]["id"]

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv")],
            title="Экспорт в CSV",
        )
        if not filename:
            return

        try:
            if current_id:
                # Полный экспорт через results_manager (с метаданными в заголовке)
                results_manager.export_to_csv(current_id, filename)
            else:
                # Резервный вариант: текущие результаты без метаданных
                flat = [_flatten_result(r) for r in self.current_results]
                with open(filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=flat[0].keys())
                    writer.writeheader()
                    writer.writerows(flat)

            messagebox.showinfo("✓ CSV экспортирован", f"Файл: {filename}")
        except Exception as e:
            messagebox.showerror("✗ Ошибка", f"Ошибка экспорта CSV: {e}")

    def compare_runs(self) -> None:
        """
        Открывает диалог выбора прогонов для сравнения на одном BER-графике.
        Пользователь выбирает любые 2+ прогона из истории.
        """
        records = results_manager.get_results_list()
        if len(records) < 2:
            messagebox.showwarning(
                "Недостаточно данных",
                "Для сравнения нужно минимум 2 сохранённых прогона.\n"
                "Сначала сохраните результаты через кнопку «Сохранить»."
            )
            return

        # ── Диалог выбора ────────────────────────────────────────────────────
        dlg = tk.Toplevel(self.root)
        dlg.title("Сравнить прогоны")
        dlg.geometry("520x420")
        dlg.configure(bg=DARK_BG)
        dlg.resizable(False, True)
        dlg.grab_set()  # модальный

        tk.Label(dlg, text="Выберите прогоны для сравнения (минимум 2, максимум 5):",
                 bg=DARK_BG, fg=TEXT_COLOR, font=("Segoe UI", 10)).pack(
            padx=15, pady=(15, 5), anchor=tk.W)

        # Список с чекбоксами
        list_frame = tk.Frame(dlg, bg=DARK_BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        canvas = tk.Canvas(list_frame, bg=DARK_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=DARK_BG)

        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        check_vars: list[tk.BooleanVar] = []
        for rec in records:
            var = tk.BooleanVar()
            check_vars.append(var)
            mod  = rec.get("modulation", "?")
            cod  = rec.get("coding_type", "нет")
            name = rec.get("name", rec["id"])
            label_text = f"{name}  [{mod} | {cod}]"
            tk.Checkbutton(
                inner, text=label_text, variable=var,
                bg=DARK_BG, fg=TEXT_COLOR, selectcolor=PANEL_BG,
                activebackground=DARK_BG, activeforeground=ACCENT_CYAN,
                font=("Segoe UI", 9), anchor=tk.W,
            ).pack(fill=tk.X, pady=1)

        # Счётчик выбранных
        count_var = tk.StringVar(value="Выбрано: 0")
        tk.Label(dlg, textvariable=count_var, bg=DARK_BG,
                 fg=ACCENT_YELLOW, font=("Segoe UI", 9)).pack(pady=(0, 5))

        def _update_count(*_):
            n = sum(v.get() for v in check_vars)
            count_var.set(f"Выбрано: {n}")
            compare_btn.config(state=tk.NORMAL if 2 <= n <= 5 else tk.DISABLED)

        for v in check_vars:
            v.trace_add("write", _update_count)

        # ── Кнопки ───────────────────────────────────────────────────────────
        btn_frame = tk.Frame(dlg, bg=DARK_BG)
        btn_frame.pack(pady=10)

        def _do_compare():
            selected_idx = [i for i, v in enumerate(check_vars) if v.get()]
            if not (2 <= len(selected_idx) <= 5):
                return
            dlg.destroy()

            try:
                results_data: list[list[dict]] = []
                labels: list[str] = []
                for idx in selected_idx:
                    rec = records[idx]
                    _, pts = results_manager.load_results(rec["id"])
                    results_data.append(pts)
                    mod  = rec.get("modulation", "?")
                    cod  = rec.get("coding_type", "нет")
                    labels.append(f"{mod} | {cod} | {rec.get('name', rec['id'])}")

                fname, fig = plot_comparison(
                    results_data, labels, output_dir="simulation_results"
                )
                if fig is not None:
                    self._current_fig = fig
                    self.display_plot(fig)
                if fname:
                    messagebox.showinfo("✓ График сохранён", f"Файл: {fname}")
            except Exception as e:
                messagebox.showerror("✗ Ошибка", f"Ошибка сравнения: {e}")

        compare_btn = ttk.Button(btn_frame, text="📈 Сравнить",
                                  command=_do_compare, state=tk.DISABLED)
        compare_btn.pack(side=tk.LEFT, padx=8)
        ttk.Button(btn_frame, text="Отмена", command=dlg.destroy).pack(side=tk.LEFT, padx=8)

    def show_log(self) -> None:
        log_dir = "logs"
        lf: Path | None = None
        if os.path.exists(log_dir):
            files = sorted(Path(log_dir).glob("sim_*.log"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if files:
                lf = files[0]
        if lf is None or not lf.exists():
            messagebox.showinfo("ℹ Информация", "Логи ещё не созданы")
            return
        lw = tk.Toplevel(self.root)
        lw.title("Логи симуляции")
        lw.geometry("700x500")
        lw.configure(bg=DARK_BG)
        txt = scrolledtext.ScrolledText(lw, bg="#1a1a2e", fg=ACCENT_GREEN,
                                         font=("Courier", 9), wrap=tk.WORD)
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        txt.insert(tk.END, lf.read_text(encoding="utf-8"))
        txt.config(state=tk.DISABLED)


# ── Точка входа ───────────────────────────────────────────────────────────────

def main() -> None:
    root = tk.Tk()
    try:
        SimulationGUI(root)
        root.mainloop()
    except Exception as e:
        import traceback
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()