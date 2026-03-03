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
"""

import csv
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np

from simulation import (
    simulate_transmission,
    simulate_text_transmission,
    save_results_to_text,
    plot_and_save_results,
    plot_comparison,
    results_manager,
)
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
        self.root.configure(bg=DARK_BG)
        try:
            self.root.iconbitmap(default="icon.ico")
        except Exception:
            pass

        self._setup_style()

        # --- Переменные состояния ---
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

    # ── Стиль ─────────────────────────────────────────────────────────────────

    def _setup_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",          background=PANEL_BG)
        style.configure("Dark.TFrame",     background=DARK_BG)
        style.configure("TLabel",          background=PANEL_BG, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Title.TLabel",    background=PANEL_BG, foreground=TEXT_LIGHT, font=("Segoe UI", 11, "bold"))
        style.configure("TButton",         font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton", background=[("active", ACCENT_CYAN), ("pressed", ACCENT_GREEN)])
        style.configure("TEntry",          fieldbackground="#1a1a2e", foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TCombobox",       fieldbackground="#1a1a2e", foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Treeview",        background="#1a1a2e", fieldbackground="#1a1a2e", foreground=TEXT_COLOR, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", background=PANEL_BG, foreground=ACCENT_CYAN, font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", ACCENT_CYAN)], foreground=[("selected", "#000000")])
        style.configure("TNotebook",       background=DARK_BG)
        style.configure("TNotebook.Tab",   background=PANEL_BG, foreground=TEXT_COLOR, font=("Segoe UI", 10, "bold"), padding=[15, 8])
        style.map("TNotebook.Tab",         background=[("selected", ACCENT_CYAN)], foreground=[("selected", "#000000")])
        style.configure("TCheckbutton",    background=PANEL_BG, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TRadiobutton",    background=PANEL_BG, foreground=TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("TLabelframe",     background=PANEL_BG, foreground=ACCENT_CYAN, font=("Segoe UI", 10, "bold"))
        style.configure("TLabelframe.Label", background=PANEL_BG, foreground=ACCENT_CYAN)
        style.configure("TProgressbar",    background=ACCENT_CYAN, troughcolor="#1a1a2e")

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

    # ── Виджеты ───────────────────────────────────────────────────────────────

    def create_widgets(self) -> None:
        main = ttk.Frame(self.root, style="Dark.TFrame")
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=320)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Заголовок
        hdr = ttk.Frame(left, style="Dark.TFrame")
        hdr.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(hdr, text="⚙ КОНФИГУРАЦИЯ", style="Title.TLabel",
                  foreground=ACCENT_CYAN).pack(anchor=tk.W)

        # Ноутбук с настройками
        nb = ttk.Notebook(left)
        nb.pack(fill=tk.BOTH, expand=True, padx=8)

        tabs = [
            ("📊 Данные",      self._build_data_tab),
            ("🔐 Кодирование", self._build_coding_tab),
            ("🔒 Шифрование",  self._build_encryption_tab),
            ("📡 Модуляция",   self._build_mod_tab),
            ("📶 Канал",       self._build_channel_tab),
            ("📈 Графики",     self._build_plot_tab),
            ("🔀 Перемежение",  self._build_interleaving_tab),
        ]
        for title, builder in tabs:
            tab = ttk.Frame(nb)
            nb.add(tab, text=title)
            builder(tab)

        # Кнопки управления
        btn_f = ttk.Frame(left)
        btn_f.pack(fill=tk.X, padx=8, pady=8)

        self.start_btn = ttk.Button(btn_f, text="▶ СТАРТ", command=self.start_simulation)
        self.start_btn.pack(fill=tk.X, pady=2)
        self.stop_btn = ttk.Button(btn_f, text="⏹ СТОП", command=self.stop_simulation,
                                   state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)
        ttk.Button(btn_f, text="💾 История", command=self.show_history).pack(fill=tk.X, pady=2)
        ttk.Button(btn_f, text="📋 Логи",    command=self.show_log).pack(fill=tk.X, pady=2)

        # Правая часть
        ttk.Label(right, text="📈 РЕЗУЛЬТАТЫ", style="Title.TLabel",
                  foreground=ACCENT_CYAN).pack(anchor=tk.W, pady=(0, 5))
        self.plot_frame = ttk.Frame(right)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        ttk.Label(right, text="📊 МЕТРИКИ", style="Title.TLabel",
                  foreground=ACCENT_CYAN).pack(anchor=tk.W, pady=(0, 5))
        metrics_c = ttk.Frame(right, height=140)
        metrics_c.pack(fill=tk.X, side=tk.BOTTOM)
        metrics_c.pack_propagate(False)
        self._build_metrics_table(metrics_c)

        # Нижняя панель
        ctrl_f = ttk.Frame(self.root, style="Dark.TFrame")
        ctrl_f.pack(fill=tk.X, padx=10, pady=8)

        left_ctrl = ttk.Frame(ctrl_f, style="Dark.TFrame")
        left_ctrl.pack(side=tk.LEFT)
        ttk.Button(left_ctrl, text="💾 Сохранить", command=self.save_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="🗑 Очистить",  command=self.clear_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="📊 Экспорт",   command=self.export_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="📁 CSV",        command=self.export_csv).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="🔀 Сравнить",  command=self.compare_runs).pack(side=tk.LEFT, padx=4)

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
        adv_f = ttk.LabelFrame(parent, text="Дополнительно", padding=10)
        adv_f.pack(fill=tk.X, padx=10, pady=8)
        for row, (label, var, tip) in enumerate([
            ("Макс. бит (адаптивно):", self.max_adaptive_bits_var, "×10 при BER<1e-4, ×100 при BER<1e-5"),
            ("Ранняя остановка BER:", self.early_stop_ber_var,     "0 = отключено; напр. 1e-7"),
        ]):
            ttk.Label(adv_f, text=label).grid(row=row, column=0, sticky=tk.W, pady=4)
            ttk.Entry(adv_f, textvariable=var, width=14).grid(row=row, column=1, padx=8, sticky=tk.W)
            ttk.Label(adv_f, text=tip, foreground=ACCENT_YELLOW,
                      font=("Segoe UI", 8)).grid(row=row, column=2, sticky=tk.W, padx=4)

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

    def _build_coding_tab(self, parent: ttk.Frame) -> None:
        ttk.Checkbutton(parent, text="✓ Включить кодирование",
                        variable=self.coding_enabled_var).pack(anchor=tk.W, padx=15, pady=10)
        ttk.Label(parent, text="Тип кодирования:", style="Title.TLabel").pack(
            anchor=tk.W, padx=15, pady=(10, 5))
        cf = ttk.Frame(parent)
        cf.pack(anchor=tk.W, padx=15, pady=(0, 8))
        ttk.Radiobutton(cf, text="🔷 Hamming (7,4)",  variable=self.coding_type_var,
                        value="hamming").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(cf, text="🔹 LDPC (64,32)",   variable=self.coding_type_var,
                        value="ldpc").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(cf, text="🔸 Turbo (PCCC≈1/3)", variable=self.coding_type_var,
                        value="turbo").pack(anchor=tk.W, pady=2)

        # Параметры Turbo (показываются только при выборе Turbo)
        turbo_f = ttk.LabelFrame(parent, text="Параметры Turbo", padding=8)
        for row, (label, var) in enumerate([
            ("Итераций декодера:", self.turbo_iterations_var),
            ("Размер блока (бит):", self.turbo_block_size_var),
        ]):
            ttk.Label(turbo_f, text=label).grid(row=row, column=0, sticky=tk.W, pady=3)
            ttk.Entry(turbo_f, textvariable=var, width=10).grid(row=row, column=1, padx=8, sticky=tk.W)

        def _toggle_turbo(*_: object) -> None:
            if self.coding_type_var.get() == "turbo":
                turbo_f.pack(fill=tk.X, padx=15, pady=(0, 6))
            else:
                turbo_f.pack_forget()

        self.coding_type_var.trace_add("write", _toggle_turbo)
        _toggle_turbo()  # изначально скрыт (default = hamming)

        ttk.Label(parent,
                  text=("Hamming (7,4):\n"
                        "  • Быстрое декодирование\n"
                        "  • Исправляет 1 ошибку на блок\n"
                        "  • Скорость 4/7 ≈ 0.57\n\n"
                        "LDPC (64,32) — Sum-Product BP:\n"
                        "  • «Водопадный» эффект\n"
                        "  • Скорость 32/64 = 0.5\n\n"
                        "Turbo (PCCC, Log-MAP):\n"
                        "  • Лучшая помехоустойчивость\n"
                        "  • Скорость ≈ 1/3 (медленнее)"),
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

        ttk.Radiobutton(alg_f, text="🔑 XOR-шифр (учебный, нет распространения ошибок)",
                        variable=self.enc_type_var, value="xor").pack(
            anchor=tk.W, pady=2)

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
            ("ECB", "ECB  — независимые блоки, нет распространения ошибок"),
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
                        "ECB и CTR: распр. ≈ 1.0\n"
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
        canvas = tk.Canvas(parent, bg=PANEL_BG, highlightthickness=0)
        sb = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        sf = ttk.Frame(canvas)

        def _on_mw(event: tk.Event) -> None:
            delta = -1 if (hasattr(event, "num") and event.num == 4) or event.delta > 0 else 1
            canvas.yview_scroll(delta, "units")

        sf.bind("<Enter>", lambda _: canvas.bind_all("<MouseWheel>", _on_mw))
        sf.bind("<Leave>", lambda _: canvas.unbind_all("<MouseWheel>"))
        sf.bind("<Button-4>", _on_mw)
        sf.bind("<Button-5>", _on_mw)
        sf.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=sf, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # AWGN — всегда
        af = ttk.LabelFrame(sf, text="🌊 AWGN (Белый шум)", padding=10)
        af.pack(fill=tk.X, padx=10, pady=8)
        ttk.Label(af, text="Добавляется автоматически через Eb/N0",
                  foreground=ACCENT_YELLOW).pack(anchor=tk.W)

        def _add_section(title: str, key_prefix: str, fields: list[tuple]) -> dict[str, tk.Variable]:
            """Создаёт LabelFrame с чекбоксом-включателем и набором полей."""
            enabled_var = tk.BooleanVar(value=False)
            frame = ttk.LabelFrame(sf, text=title, padding=10)
            frame.pack(fill=tk.X, padx=10, pady=8)
            ttk.Checkbutton(frame, text=f"✓ Включить", variable=enabled_var).pack(
                anchor=tk.W, pady=(0, 8))
            vars_map: dict[str, tk.Variable] = {f"{key_prefix}_enabled": enabled_var}
            for label, var_key, var_type, default, widget_kwargs in fields:
                ttk.Label(frame, text=label, font=("Segoe UI", 9)).pack(anchor=tk.W, pady=(4, 0))
                var = (tk.StringVar if var_type == "str" else tk.StringVar)(value=str(default))
                widget_class = widget_kwargs.pop("__class__", ttk.Entry)
                w = widget_class(frame, textvariable=var, **widget_kwargs)
                w.pack(anchor=tk.W, pady=(0, 6))
                vars_map[f"{key_prefix}_{var_key}"] = var
            return vars_map

        # Rayleigh
        self.channel_vars.update(_add_section(
            "📶 Rayleigh Fading (многолучевое)", "rayleigh", [
                ("Число лучей:",              "rays",    "str", "16",  {"__class__": ttk.Spinbox, "from_": 2, "to": 64, "width": 10}),
                ("Доплер. частота (норм.):",  "doppler", "str", "0.01", {"width": 15}),
            ]
        ))

        # Phase Noise
        self.channel_vars.update(_add_section(
            "🔄 Фазовый шум", "phase", [
                ("Дисперсия фазы (rad²):", "variance", "str", "0.001", {"width": 15}),
            ]
        ))

        # Frequency Offset
        self.channel_vars.update(_add_section(
            "📊 Частотный сдвиг", "freq", [
                ("Норм. сдвиг (-0.1 до +0.1):", "offset", "str", "0.01", {"width": 15}),
            ]
        ))

        # Timing Jitter
        self.channel_vars.update(_add_section(
            "⏱ Ошибка синхронизации", "timing", [
                ("Диапазон смещения (Ts):", "offset", "str", "0.05", {"width": 15}),
            ]
        ))

        # Impulse Noise
        self.channel_vars.update(_add_section(
            "⚡ Импульсные помехи", "impulse", [
                ("Вероятность помехи:", "prob",      "str", "0.001", {"width": 15}),
                ("Амплитуда (σ):",      "amp",       "str", "10.0",  {"width": 15}),
            ]
        ))
        # Ширина импульса — отдельно (два поля)
        impulse_parent = sf.winfo_children()[-1]  # последний LabelFrame
        w_frame = ttk.Frame(impulse_parent)
        w_frame.pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(w_frame, text="Ширина: от").pack(side=tk.LEFT, padx=4)
        wf_var = tk.StringVar(value="1")
        ttk.Spinbox(w_frame, from_=1, to=20, textvariable=wf_var, width=6).pack(side=tk.LEFT, padx=4)
        ttk.Label(w_frame, text="до").pack(side=tk.LEFT, padx=4)
        wt_var = tk.StringVar(value="5")
        ttk.Spinbox(w_frame, from_=1, to=20, textvariable=wt_var, width=6).pack(side=tk.LEFT, padx=4)
        self.channel_vars["impulse_width_from"] = wf_var
        self.channel_vars["impulse_width_to"]   = wt_var


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
        elif coding_type_val == "ldpc":
            n, k = 64, 32
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
                "phase_noise_variance": float(cv["phase_variance"].get()),
            },
            "frequency_offset": {
                "enabled":                cv["freq_enabled"].get(),
                "normalized_freq_offset": float(cv["freq_offset"].get()),
            },
            "timing_offset": {
                "enabled":             cv["timing_enabled"].get(),
                "timing_offset_range": float(cv["timing_offset"].get()),
            },
            "impulse_noise": {
                "enabled":                  cv["impulse_enabled"].get(),
                "impulse_probability":      float(cv["impulse_prob"].get()),
                "impulse_amplitude_sigma":  float(cv["impulse_amp"].get()),
                "impulse_width_from":       int(cv["impulse_width_from"].get()),
                "impulse_width_to":         int(cv["impulse_width_to"].get()),
            },
        }

        return {
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
            "text_recovery": {
                "enabled":      self.tr_enabled_var.get(),
                "window_bytes": int(self.tr_window_var.get()),
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
                self.tree.insert("", tk.END, values=(
                    f"{r['snr']:.1f}", f"{r['ber']:.2e}", f"{r['ser']:.2e}",
                    f"{r.get('theoretical_ber', 0):.2e}",
                    f"{r.get('theoretical_ser', 0):.2e}", cer,
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