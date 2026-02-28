import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import numpy as np
import time
import os
import logging
import json
import csv
from pathlib import Path
from datetime import datetime

from simulation import (
    simulate_transmission, simulate_text_transmission,
    save_results_to_text, plot_and_save_results, results_manager
)

# Константы дизайна
DARK_BG = '#0f0f23'
PANEL_BG = '#16213e'
ACCENT_CYAN = '#00d9ff'
ACCENT_PINK = '#ff006e'
ACCENT_GREEN = '#39ff14'
ACCENT_YELLOW = '#ffbe0b'
TEXT_COLOR = '#e0e0e0'
TEXT_LIGHT = '#ffffff'


class SimulationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Симулятор цифровой связи — M-PSK / M-QAM")
        self.root.geometry("1400x900")
        self.root.configure(bg=DARK_BG)

        # Иконка окна (если существует)
        try:
            self.root.iconbitmap(default='icon.ico')
        except:
            pass

        self.style = self._setup_style()

        # Переменные состояния
        self.mode_var = tk.StringVar(value="random")
        self.num_bits_var = tk.StringVar(value="100000")
        self.num_simulations_var = tk.StringVar(value="5")
        self.text_file_var = tk.StringVar(value="input.txt")
        self.text_encoding_var = tk.StringVar(value="utf-8")
        self.max_text_length_var = tk.StringVar(value="10000")
        self.num_repetitions_var = tk.StringVar(value="3")

        self.modulation_type_var = tk.StringVar(value="PSK")
        self.modulation_order_var = tk.StringVar(value="8")
        self.gray_code_var = tk.BooleanVar(value=True)

        self.coding_enabled_var = tk.BooleanVar(value=True)
        self.coding_type_var = tk.StringVar(value="hamming")

        self.ebn0_start_var = tk.StringVar(value="0")
        self.ebn0_stop_var = tk.StringVar(value="10")
        self.ebn0_step_var = tk.StringVar(value="1")
        self.show_theo_var = tk.BooleanVar(value=True)

        # Queue и progress
        self.queue = queue.Queue()
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_label_var = tk.StringVar(value="Готов")
        self.eta_var = tk.StringVar(value="")

        self.running = False
        self.current_results = None
        self.current_config = None
        self._current_canvas = None
        self._current_fig = None

        # Переменные канала (инициализируются в методе)
        self.channel_vars = {}

        self.create_widgets()
        self.root.after(100, self.process_queue)

    def _setup_style(self):
        """Настраивает стиль ttk"""
        style = ttk.Style()
        style.theme_use('clam')

        # Основные цвета
        style.configure('TFrame', background=PANEL_BG)
        style.configure('Dark.TFrame', background=DARK_BG)
        style.configure('TLabel', background=PANEL_BG, foreground=TEXT_COLOR,
                        font=('Segoe UI', 10))
        style.configure('Title.TLabel', background=PANEL_BG, foreground=TEXT_LIGHT,
                        font=('Segoe UI', 11, 'bold'))
        style.configure('TButton', font=('Segoe UI', 10, 'bold'), padding=6)
        style.map('TButton',
                  background=[('active', ACCENT_CYAN), ('pressed', ACCENT_GREEN)])

        style.configure('TEntry', fieldbackground='#1a1a2e', foreground=TEXT_COLOR,
                        font=('Segoe UI', 10))
        style.configure('TCombobox', fieldbackground='#1a1a2e', foreground=TEXT_COLOR,
                        font=('Segoe UI', 10))
        style.configure('Treeview', background='#1a1a2e', fieldbackground='#1a1a2e',
                        foreground=TEXT_COLOR, font=('Segoe UI', 9))
        style.configure('Treeview.Heading', background=PANEL_BG, foreground=ACCENT_CYAN,
                        font=('Segoe UI', 10, 'bold'))
        style.map('Treeview', background=[('selected', ACCENT_CYAN)],
                  foreground=[('selected', '#000000')])

        style.configure('TNotebook', background=DARK_BG)
        style.configure('TNotebook.Tab', background=PANEL_BG, foreground=TEXT_COLOR,
                        font=('Segoe UI', 10, 'bold'), padding=[15, 8])
        style.map('TNotebook.Tab',
                  background=[('selected', ACCENT_CYAN)],
                  foreground=[('selected', '#000000')])

        style.configure('TCheckbutton', background=PANEL_BG, foreground=TEXT_COLOR,
                        font=('Segoe UI', 10))
        style.configure('TRadiobutton', background=PANEL_BG, foreground=TEXT_COLOR,
                        font=('Segoe UI', 10))
        style.configure('TLabelframe', background=PANEL_BG, foreground=ACCENT_CYAN,
                        font=('Segoe UI', 10, 'bold'))
        style.configure('TLabelframe.Label', background=PANEL_BG, foreground=ACCENT_CYAN)

        # Progressbar
        style.configure('TProgressbar', background=ACCENT_CYAN, troughcolor='#1a1a2e')

        return style

    def create_widgets(self):
        """Создает главный интерфейс"""
        # Главный контейнер
        main_container = ttk.Frame(self.root, style='Dark.TFrame')
        main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        # Левая панель (конфигурация)
        left_panel = ttk.Frame(main_container, width=320)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=0, pady=0)
        left_panel.pack_propagate(False)

        # Правая панель (графики и таблица)
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Заголовок слева
        header = ttk.Frame(left_panel, style='Dark.TFrame')
        header.pack(fill=tk.X, padx=10, pady=10)

        title = ttk.Label(header, text="⚙ КОНФИГУРАЦИЯ", style='Title.TLabel',
                          foreground=ACCENT_CYAN)
        title.pack(anchor=tk.W)

        # Notebook с табами
        notebook = ttk.Notebook(left_panel)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=0)

        # Табы конфигурации
        data_tab = ttk.Frame(notebook)
        notebook.add(data_tab, text="📊 Данные")
        self._build_data_tab(data_tab)

        coding_tab = ttk.Frame(notebook)
        notebook.add(coding_tab, text="🔐 Кодирование")
        self._build_coding_tab(coding_tab)

        mod_tab = ttk.Frame(notebook)
        notebook.add(mod_tab, text="📡 Модуляция")
        self._build_mod_tab(mod_tab)

        channel_tab = ttk.Frame(notebook)
        notebook.add(channel_tab, text="📶 Канал")
        self._build_channel_tab(channel_tab)

        plot_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text="📈 Графики")
        self._build_plot_tab(plot_tab)

        # Разделитель
        sep = ttk.Frame(left_panel, height=2)
        sep.pack(fill=tk.X, padx=8, pady=8)

        # Кнопка управления
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, padx=8, pady=4)

        self.start_btn = ttk.Button(btn_frame, text="▶ СТАРТ", command=self.start_simulation)
        self.start_btn.pack(fill=tk.X, pady=2)

        self.stop_btn = ttk.Button(btn_frame, text="⏹ СТОП", command=self.stop_simulation,
                                   state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=2)

        ttk.Button(btn_frame, text="💾 История", command=self.show_history).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="📋 Логи", command=self.show_log).pack(fill=tk.X, pady=2)

        # Правая часть: график
        plot_label = ttk.Label(right_panel, text="📈 РЕЗУЛЬТАТЫ",
                               style='Title.TLabel', foreground=ACCENT_CYAN)
        plot_label.pack(anchor=tk.W, pady=(0, 5))

        self.plot_frame = ttk.Frame(right_panel)
        self.plot_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        # Таблица метрик
        metrics_label = ttk.Label(right_panel, text="📊 МЕТРИКИ",
                                  style='Title.TLabel', foreground=ACCENT_CYAN)
        metrics_label.pack(anchor=tk.W, pady=(0, 5))

        metrics_container = ttk.Frame(right_panel, height=140)
        metrics_container.pack(fill=tk.X, side=tk.BOTTOM)
        metrics_container.pack_propagate(False)
        self._build_metrics_table(metrics_container)

        # Нижняя панель управления и progress
        control_frame = ttk.Frame(self.root, style='Dark.TFrame')
        control_frame.pack(fill=tk.X, padx=10, pady=8)

        left_ctrl = ttk.Frame(control_frame, style='Dark.TFrame')
        left_ctrl.pack(side=tk.LEFT)

        ttk.Button(left_ctrl, text="💾 Сохранить",
                   command=self.save_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="🗑 Очистить",
                   command=self.clear_results).pack(side=tk.LEFT, padx=4)
        ttk.Button(left_ctrl, text="📊 Экспорт",
                   command=self.export_results).pack(side=tk.LEFT, padx=4)

        right_ctrl = ttk.Frame(control_frame, style='Dark.TFrame')
        right_ctrl.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))

        prog_info = ttk.Frame(right_ctrl, style='Dark.TFrame')
        prog_info.pack(side=tk.LEFT, padx=20)

        self.progress_label = ttk.Label(prog_info, textvariable=self.progress_label_var,
                                        foreground=ACCENT_GREEN, style='Title.TLabel')
        self.progress_label.pack(anchor=tk.W)

        self.eta_label = ttk.Label(prog_info, textvariable=self.eta_var,
                                   foreground=ACCENT_YELLOW, font=('Segoe UI', 9))
        self.eta_label.pack(anchor=tk.W)

        self.progress = ttk.Progressbar(right_ctrl, variable=self.progress_var,
                                        length=400, mode='determinate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True)

    def _build_data_tab(self, parent):
        """Таб конфигурации данных"""
        ttk.Label(parent, text="Режим генерации:", style='Title.TLabel').pack(anchor=tk.W,
                                                                              padx=10, pady=(10, 5))
        mode_frame = ttk.Frame(parent)
        mode_frame.pack(anchor=tk.W, padx=10, pady=(0, 15))
        ttk.Radiobutton(mode_frame, text="📊 Случайные биты",
                        variable=self.mode_var, value="random").pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(mode_frame, text="📝 Текст",
                        variable=self.mode_var, value="text").pack(side=tk.LEFT, padx=8)

        # Случайные биты
        rand_frame = ttk.LabelFrame(parent, text="Параметры случайных бит", padding=10)
        rand_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(rand_frame, text="Количество бит:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(rand_frame, textvariable=self.num_bits_var, width=15).grid(row=0, column=1,
                                                                             padx=10, sticky=tk.W)

        ttk.Label(rand_frame, text="Число усреднений:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(rand_frame, textvariable=self.num_simulations_var, width=15).grid(row=1,
                                                                                    column=1, padx=10, sticky=tk.W)

        # Текст
        text_frame = ttk.LabelFrame(parent, text="Параметры текста", padding=10)
        text_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(text_frame, text="Файл:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(text_frame, textvariable=self.text_file_var, width=15).grid(row=0, column=1,
                                                                              padx=10, sticky=tk.W)
        ttk.Button(text_frame, text="📁", width=3,
                   command=self.browse_file).grid(row=0, column=2, padx=5)

        ttk.Label(text_frame, text="Кодировка:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(text_frame, values=['utf-8', 'utf-16', 'ascii', 'cp1251'],
                     textvariable=self.text_encoding_var, width=12, state='readonly').grid(row=1,
                                                                                           column=1, padx=10,
                                                                                           sticky=tk.W)

        ttk.Label(text_frame, text="Макс. длина (симв):").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(text_frame, textvariable=self.max_text_length_var, width=15).grid(row=2,
                                                                                    column=1, padx=10, sticky=tk.W)

        ttk.Label(text_frame, text="Повторений:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(text_frame, textvariable=self.num_repetitions_var, width=15).grid(row=3,
                                                                                    column=1, padx=10, sticky=tk.W)

        # Диапазон Eb/N0
        ttk.Label(parent, text="Диапазон Eb/N0 (дБ):", style='Title.TLabel').pack(anchor=tk.W,
                                                                                  padx=10, pady=(15, 5))
        range_f = ttk.Frame(parent)
        range_f.pack(anchor=tk.W, padx=10, pady=(0, 10))

        ttk.Label(range_f, text="От:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(range_f, textvariable=self.ebn0_start_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(range_f, text="До:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(range_f, textvariable=self.ebn0_stop_var, width=6).pack(side=tk.LEFT, padx=5)
        ttk.Label(range_f, text="Шаг:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(range_f, textvariable=self.ebn0_step_var, width=6).pack(side=tk.LEFT, padx=5)

    def _build_coding_tab(self, parent):
        """Таб конфигурации кодирования"""
        ttk.Checkbutton(parent, text="✓ Включить кодирование",
                        variable=self.coding_enabled_var).pack(anchor=tk.W, padx=15, pady=10)

        ttk.Label(parent, text="Тип кодирования:", style='Title.TLabel').pack(anchor=tk.W,
                                                                              padx=15, pady=(10, 5))
        coding_frame = ttk.Frame(parent)
        coding_frame.pack(anchor=tk.W, padx=15, pady=(0, 20))

        ttk.Radiobutton(coding_frame, text="🔷 Hamming (7,4)",
                        variable=self.coding_type_var, value="hamming").pack(side=tk.LEFT,
                                                                             padx=10, pady=5)
        ttk.Radiobutton(coding_frame, text="🔹 LDPC (12,6)",
                        variable=self.coding_type_var, value="ldpc").pack(side=tk.LEFT, padx=10, pady=5)

        info = ("Hamming (7,4):\n"
                "  • Быстрое декодирование\n"
                "  • Исправляет 1 ошибку\n"
                "  • Скорость 4/7 ≈ 0.57\n\n"
                "LDPC (12,6):\n"
                "  • Лучше при высоком шуме\n"
                "  • Скорость 6/12 = 0.5")
        ttk.Label(parent, text=info, justify=tk.LEFT,
                  foreground=ACCENT_YELLOW).pack(anchor=tk.W, padx=15, pady=10)

    def _build_mod_tab(self, parent):
        """Таб конфигурации модуляции"""
        ttk.Label(parent, text="Тип модуляции:", style='Title.TLabel').pack(anchor=tk.W,
                                                                            padx=15, pady=(10, 5))
        mod_type_frame = ttk.Frame(parent)
        mod_type_frame.pack(anchor=tk.W, padx=15, pady=(0, 20))

        ttk.Radiobutton(mod_type_frame, text="📡 PSK",
                        variable=self.modulation_type_var, value="PSK").pack(side=tk.LEFT,
                                                                             padx=10, pady=5)
        ttk.Radiobutton(mod_type_frame, text="📡 QAM",
                        variable=self.modulation_type_var, value="QAM").pack(side=tk.LEFT,
                                                                             padx=10, pady=5)

        ttk.Label(parent, text="Порядок модуляции (M):", style='Title.TLabel').pack(anchor=tk.W,
                                                                                    padx=15, pady=(10, 5))

        self.mod_order_combo = ttk.Combobox(parent, textvariable=self.modulation_order_var,
                                            values=['4', '16', '64'] if self.modulation_type_var.get() == 'QAM' else [
                                                '2', '4', '8', '16'],
                                            state='readonly', width=10, font=('Segoe UI', 10, 'bold'))
        self.mod_order_combo.pack(anchor=tk.W, padx=15, pady=(0, 20))

        def update_mod_order(*args):
            if self.modulation_type_var.get() == 'QAM':
                self.mod_order_combo['values'] = ('4', '16', '64')
                if self.modulation_order_var.get() not in ('4', '16', '64'):
                    self.modulation_order_var.set('16')
            else:
                self.mod_order_combo['values'] = ('2', '4', '8', '16')
                if self.modulation_order_var.get() not in ('2', '4', '8', '16'):
                    self.modulation_order_var.set('8')

        self.modulation_type_var.trace_add('write', update_mod_order)

        ttk.Checkbutton(parent, text="✓ Код Грея",
                        variable=self.gray_code_var).pack(anchor=tk.W, padx=15, pady=(0, 20))

        info = ("PSK: фазовая манипуляция\n"
                "  • M = 2, 4, 8, 16\n"
                "  • Оптимален для огран. полосы\n\n"
                "QAM: амплитудно-фазовая\n"
                "  • M = 4, 16, 64\n"
                "  • Выше спектр. эффективность")
        ttk.Label(parent, text=info, justify=tk.LEFT,
                  foreground=ACCENT_YELLOW).pack(anchor=tk.W, padx=15, pady=10)

    def _build_channel_tab(self, parent):
        """Таб конфигурации моделей канала"""
        # Создаем фрейм для скролла
        canvas = tk.Canvas(parent, bg=PANEL_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # --- mousewheel scrolling ---
        def _on_channel_mousewheel(event):
            if hasattr(event, 'num') and event.num in (4, 5):  # Linux
                canvas.yview_scroll(-1 if event.num == 4 else 1, "units")
            else:  # Windows / Mac
                canvas.yview_scroll(int(-event.delta / 120), "units")

        scrollable_frame.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_channel_mousewheel))
        scrollable_frame.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        scrollable_frame.bind("<Button-4>", _on_channel_mousewheel)
        scrollable_frame.bind("<Button-5>", _on_channel_mousewheel)
        # --- end scrolling ---

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # AWGN (всегда включен)
        awgn_frame = ttk.LabelFrame(scrollable_frame, text="🌊 AWGN (Белый шум)", padding=10)
        awgn_frame.pack(fill=tk.X, padx=10, pady=8)
        ttk.Label(awgn_frame, text="Добавляется автоматически через Eb/N0",
                  foreground=ACCENT_YELLOW).pack(anchor=tk.W)

        # Rayleigh Fading
        rayleigh_var = tk.BooleanVar(value=False)
        rayleigh_frame = ttk.LabelFrame(scrollable_frame, text="📶 Rayleigh Fading (многолучевое)", padding=10)
        rayleigh_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Checkbutton(rayleigh_frame, text="✓ Включить Rayleigh",
                        variable=rayleigh_var).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(rayleigh_frame, text="Число лучей:", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        rayleigh_rays_var = tk.StringVar(value="4")
        ttk.Spinbox(rayleigh_frame, from_=2, to=16, textvariable=rayleigh_rays_var,
                    width=10).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(rayleigh_frame, text="Доплер. частота (норм.):", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        rayleigh_doppler_var = tk.StringVar(value="0.01")
        ttk.Entry(rayleigh_frame, textvariable=rayleigh_doppler_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        info_rayleigh = ("Модель: много рассеянных лучей\n"
                        "Типично: городская среда, NLOS")
        ttk.Label(rayleigh_frame, text=info_rayleigh, foreground=ACCENT_YELLOW,
                  font=('Segoe UI', 8), justify=tk.LEFT).pack(anchor=tk.W)

        # Rician Fading
        rician_var = tk.BooleanVar(value=False)
        rician_frame = ttk.LabelFrame(scrollable_frame, text="📡 Rician Fading (LOS + рассеяние)", padding=10)
        rician_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Checkbutton(rician_frame, text="✓ Включить Rician",
                        variable=rician_var).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(rician_frame, text="Число лучей:", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        rician_rays_var = tk.StringVar(value="4")
        ttk.Spinbox(rician_frame, from_=2, to=16, textvariable=rician_rays_var,
                    width=10).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(rician_frame, text="Доплер. частота (норм.):", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        rician_doppler_var = tk.StringVar(value="0.01")
        ttk.Entry(rician_frame, textvariable=rician_doppler_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(rician_frame, text="K-фактор (LOS/NLOS):", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        rician_k_var = tk.StringVar(value="3.0")
        ttk.Entry(rician_frame, textvariable=rician_k_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        info_rician = ("Модель: прямой луч + рассеяние\n"
                      "Типично: пригород, спутник\n"
                      "K → ∞: AWGN, K = 0: Rayleigh")
        ttk.Label(rician_frame, text=info_rician, foreground=ACCENT_YELLOW,
                  font=('Segoe UI', 8), justify=tk.LEFT).pack(anchor=tk.W)

        # Phase Noise
        phase_var = tk.BooleanVar(value=False)
        phase_frame = ttk.LabelFrame(scrollable_frame, text="🔄 Фазовый шум", padding=10)
        phase_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Checkbutton(phase_frame, text="✓ Включить фазовый шум",
                        variable=phase_var).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(phase_frame, text="Дисперсия фазы (rad²):", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        phase_var_var = tk.StringVar(value="0.001")
        ttk.Entry(phase_frame, textvariable=phase_var_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        info_phase = ("Источник: нестабильность генератора\n"
                     "Модель: Wiener процесс фазы")
        ttk.Label(phase_frame, text=info_phase, foreground=ACCENT_YELLOW,
                  font=('Segoe UI', 8), justify=tk.LEFT).pack(anchor=tk.W)

        # Frequency Offset
        freq_var = tk.BooleanVar(value=False)
        freq_frame = ttk.LabelFrame(scrollable_frame, text="📊 Частотный сдвиг", padding=10)
        freq_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Checkbutton(freq_frame, text="✓ Включить частотный сдвиг",
                        variable=freq_var).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(freq_frame, text="Норм. сдвиг (-0.1 до +0.1):", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        freq_offset_var = tk.StringVar(value="0.01")
        ttk.Entry(freq_frame, textvariable=freq_offset_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        info_freq = ("Причина: расстройка приемника\n"
                    "Эффект: линейный рост фазы")
        ttk.Label(freq_frame, text=info_freq, foreground=ACCENT_YELLOW,
                  font=('Segoe UI', 8), justify=tk.LEFT).pack(anchor=tk.W)

        # Timing Jitter
        timing_var = tk.BooleanVar(value=False)
        timing_frame = ttk.LabelFrame(scrollable_frame, text="⏱ Ошибка синхронизации", padding=10)
        timing_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Checkbutton(timing_frame, text="✓ Включить timing jitter",
                        variable=timing_var).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(timing_frame, text="Диапазон смещения (Ts):", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        timing_offset_var = tk.StringVar(value="0.05")
        ttk.Entry(timing_frame, textvariable=timing_offset_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        info_timing = ("Причина: неточная синхронизация\n"
                      "Эффект: случайное смещение времени")
        ttk.Label(timing_frame, text=info_timing, foreground=ACCENT_YELLOW,
                  font=('Segoe UI', 8), justify=tk.LEFT).pack(anchor=tk.W)

        # Impulse Noise
        impulse_var = tk.BooleanVar(value=False)
        impulse_frame = ttk.LabelFrame(scrollable_frame, text="⚡ Импульсные помехи", padding=10)
        impulse_frame.pack(fill=tk.X, padx=10, pady=8)

        ttk.Checkbutton(impulse_frame, text="✓ Включить импульсные помехи",
                        variable=impulse_var).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(impulse_frame, text="Вероятность помехи:", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        impulse_prob_var = tk.StringVar(value="0.001")
        ttk.Entry(impulse_frame, textvariable=impulse_prob_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(impulse_frame, text="Амплитуда (σ):", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        impulse_amp_var = tk.StringVar(value="10.0")
        ttk.Entry(impulse_frame, textvariable=impulse_amp_var, width=15).pack(anchor=tk.W, pady=(0, 10))

        ttk.Label(impulse_frame, text="Ширина импульса:", font=('Segoe UI', 9)).pack(anchor=tk.W, pady=(5, 0))
        width_frame = ttk.Frame(impulse_frame)
        width_frame.pack(anchor=tk.W, pady=(0, 10))
        ttk.Label(width_frame, text="От:").pack(side=tk.LEFT, padx=5)
        impulse_width_from_var = tk.StringVar(value="1")
        ttk.Spinbox(width_frame, from_=1, to=20, textvariable=impulse_width_from_var,
                    width=8).pack(side=tk.LEFT, padx=5)
        ttk.Label(width_frame, text="До:").pack(side=tk.LEFT, padx=5)
        impulse_width_to_var = tk.StringVar(value="5")
        ttk.Spinbox(width_frame, from_=1, to=20, textvariable=impulse_width_to_var,
                    width=8).pack(side=tk.LEFT, padx=5)

        info_impulse = ("Источник: помехи в сети\n"
                       "Модель: редкие, мощные импульсы")
        ttk.Label(impulse_frame, text=info_impulse, foreground=ACCENT_YELLOW,
                  font=('Segoe UI', 8), justify=tk.LEFT).pack(anchor=tk.W)

        # Сохраняем переменные в объект для последующего доступа
        self.channel_vars = {
            'rayleigh_enabled': rayleigh_var,
            'rayleigh_rays': rayleigh_rays_var,
            'rayleigh_doppler': rayleigh_doppler_var,
            'rician_enabled': rician_var,
            'rician_rays': rician_rays_var,
            'rician_doppler': rician_doppler_var,
            'rician_k': rician_k_var,
            'phase_enabled': phase_var,
            'phase_variance': phase_var_var,
            'freq_enabled': freq_var,
            'freq_offset': freq_offset_var,
            'timing_enabled': timing_var,
            'timing_offset': timing_offset_var,
            'impulse_enabled': impulse_var,
            'impulse_prob': impulse_prob_var,
            'impulse_amp': impulse_amp_var,
            'impulse_width_from': impulse_width_from_var,
            'impulse_width_to': impulse_width_to_var
        }

    def _build_plot_tab(self, parent):
        """Таб конфигурации графиков"""
        ttk.Checkbutton(parent, text="✓ Показывать теоретические кривые",
                        variable=self.show_theo_var).pack(anchor=tk.W, padx=15, pady=10)

        info = ("Теоретические кривые рассчиты-\n"
                "ваются на основе точных формул\n"
                "для M-PSK и M-QAM из\n"
                "Proakis 'Digital Communications'\n\n"
                "Помогают сравнить с экспери-\n"
                "ментом и оценить эффективность\n"
                "кодирования.")
        ttk.Label(parent, text=info, justify=tk.LEFT,
                  foreground=ACCENT_YELLOW).pack(anchor=tk.W, padx=15, pady=20)

    def _build_metrics_table(self, parent):
        """Таблица метрик"""
        cols = ('snr', 'ber', 'ser', 'theo_ber', 'theo_ser', 'cer')
        self.tree = ttk.Treeview(parent, columns=cols, show='headings', height=5)

        self.tree.heading('snr', text='Eb/N0\n(дБ)')
        self.tree.heading('ber', text='BER\n(эксп.)')
        self.tree.heading('ser', text='SER\n(эксп.)')
        self.tree.heading('theo_ber', text='BER\n(теор.)')
        self.tree.heading('theo_ser', text='SER\n(теор.)')
        self.tree.heading('cer', text='CER\n(%)')

        col_widths = [80, 100, 100, 110, 110, 90]
        for c, w in zip(cols, col_widths):
            self.tree.column(c, width=w, anchor=tk.CENTER)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    def browse_file(self):
        """Выбор файла текста"""
        fn = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if fn:
            self.text_file_var.set(fn)

    def build_config(self):
        """Строит конфигурацию из параметров интерфейса"""
        try:
            start = float(self.ebn0_start_var.get())
            stop = float(self.ebn0_stop_var.get())
            step = float(self.ebn0_step_var.get())
            rng = np.arange(start, stop + step / 2, step)

            if self.coding_type_var.get() == "hamming":
                n, k = 7, 4
            else:
                n, k = 12, 6

            # Конфиг канала
            channel_config = {
                'awgn': {'enabled': True},
                'rayleigh': {
                    'enabled': self.channel_vars['rayleigh_enabled'].get(),
                    'num_rays': int(self.channel_vars['rayleigh_rays'].get()),
                    'normalized_doppler': float(self.channel_vars['rayleigh_doppler'].get())
                },
                'rician': {
                    'enabled': self.channel_vars['rician_enabled'].get(),
                    'num_rays': int(self.channel_vars['rician_rays'].get()),
                    'normalized_doppler': float(self.channel_vars['rician_doppler'].get()),
                    'rician_factor_k': float(self.channel_vars['rician_k'].get())
                },
                'phase_noise': {
                    'enabled': self.channel_vars['phase_enabled'].get(),
                    'phase_noise_variance': float(self.channel_vars['phase_variance'].get())
                },
                'frequency_offset': {
                    'enabled': self.channel_vars['freq_enabled'].get(),
                    'normalized_freq_offset': float(self.channel_vars['freq_offset'].get())
                },
                'timing_offset': {
                    'enabled': self.channel_vars['timing_enabled'].get(),
                    'timing_offset_range': float(self.channel_vars['timing_offset'].get())
                },
                'impulse_noise': {
                    'enabled': self.channel_vars['impulse_enabled'].get(),
                    'impulse_probability': float(self.channel_vars['impulse_prob'].get()),
                    'impulse_amplitude_sigma': float(self.channel_vars['impulse_amp'].get()),
                    'impulse_width_from': int(self.channel_vars['impulse_width_from'].get()),
                    'impulse_width_to': int(self.channel_vars['impulse_width_to'].get())
                }
            }

            config = {
                "simulation_mode": self.mode_var.get(),
                "modulation": {
                    "type": self.modulation_type_var.get(),
                    "order": int(self.modulation_order_var.get()),
                    "use_gray_code": self.gray_code_var.get()
                },
                "coding": {
                    "enabled": self.coding_enabled_var.get(),
                    "type": self.coding_type_var.get(),
                    "n": n,
                    "k": k
                },
                "channel": channel_config,
                "text_settings": {
                    "text_file": self.text_file_var.get(),
                    "text_encoding": self.text_encoding_var.get(),
                    "max_text_length": int(self.max_text_length_var.get()),
                    "num_repetitions": int(self.num_repetitions_var.get())
                },
                "random_settings": {
                    "num_bits": int(self.num_bits_var.get()),
                    "num_simulations": int(self.num_simulations_var.get())
                },
                "ebn0_dB_range": rng.tolist()
            }
            return config
        except Exception as e:
            messagebox.showerror("Ошибка ввода", f"Неверные параметры:\n{e}")
            return None

    # ---------- ИЗМЕНЕНИЯ ЗДЕСЬ (ЛОГИРОВАНИЕ) ----------
    def start_simulation(self):
        """Запуск симуляции"""
        for r in self.tree.get_children():
            self.tree.delete(r)
        self.clear_plot()

        config = self.build_config()
        if config is None:
            return

        mode = config["simulation_mode"]

        # ----- НАСТРОЙКА ЛОГИРОВАНИЯ ДЛЯ ЭТОЙ СИМУЛЯЦИИ -----
        # Создаём папку logs, если её нет
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Генерируем имя файла с временной меткой
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = os.path.join(log_dir, f"sim_{timestamp}.log")

        # Получаем корневой логгер
        root_logger = logging.getLogger()

        # Удаляем все старые обработчики (чтобы не писать в предыдущий файл)
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Создаём новый файловый обработчик
        file_handler = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        # Добавляем также консольный обработчик (можно убрать, если не нужен)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        root_logger.setLevel(logging.INFO)

        root_logger.info(f"Лог-файл этой симуляции: {log_filename}")
        # ----------------------------------------------------

        self.running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        self.sim_thread = threading.Thread(target=self.run_simulation,
                                           args=(config, mode), daemon=True)
        self.sim_thread.start()

    def run_simulation(self, config, mode):
        """Основной цикл симуляции"""
        start_time = time.time()
        rng = config["ebn0_dB_range"]
        results = []
        total = len(rng)

        # Логируем конфигурацию в начале
        logging.info("="*60)
        logging.info("НАЧАЛО СИМУЛЯЦИИ")
        logging.info(f"Режим: {mode}")
        logging.info(f"Модуляция: {config['modulation']['type']}-{config['modulation']['order']}, Код Грея: {config['modulation']['use_gray_code']}")
        if config['coding']['enabled']:
            logging.info(f"Кодирование: {config['coding']['type']} ({config['coding']['n']},{config['coding']['k']}), скорость={config['coding']['k']/config['coding']['n']:.3f}")
        else:
            logging.info("Кодирование: отключено")
        logging.info(f"Диапазон Eb/N0: {min(rng)}..{max(rng)} дБ, шаг {rng[1]-rng[0] if len(rng)>1 else 'N/A'}")
        logging.info("="*60)

        if mode == "text":
            try:
                with open(config["text_settings"]["text_file"], 'r',
                          encoding=config["text_settings"]["text_encoding"]) as f:
                    text = f.read()
                logging.info(f"Текст загружен из файла {config['text_settings']['text_file']}, длина {len(text)} символов")
            except Exception as e:
                self.queue.put(("error", f"Ошибка чтения файла: {e}"))
                return

        for idx, snr in enumerate(rng):
            if not self.running:
                logging.warning("Симуляция прервана пользователем")
                break

            # Progress и ETA
            progress_pct = (idx / total) * 100
            elapsed = time.time() - start_time
            if idx > 0:
                avg_time_per_point = elapsed / idx
                remaining_points = total - idx
                eta_seconds = avg_time_per_point * remaining_points
                eta_str = f"ETA: {int(eta_seconds)}s" if eta_seconds < 3600 else f"ETA: {eta_seconds / 60:.1f}m"
            else:
                eta_str = "Расчет ETA..."

            self.queue.put(("progress", progress_pct,
                            f"SNR={snr:.1f} дБ ({idx + 1}/{total})", eta_str))

            if mode == "random":
                num_sim = config["random_settings"]["num_simulations"]
                stats_list = []
                for _ in range(num_sim):
                    if not self.running:
                        break
                    stats_list.append(simulate_transmission(config, snr))

                if not self.running:
                    break

                avg = {
                    "snr": snr,
                    "ber": float(np.mean([s["ber"] for s in stats_list])),
                    "ser": float(np.mean([s["ser"] for s in stats_list])),
                    "theoretical_ber": float(np.mean([s["theoretical_ber"] for s in stats_list])),
                    "theoretical_ser": float(np.mean([s["theoretical_ser"] for s in stats_list])),
                    "cer": 0
                }
            else:
                reps = config["text_settings"]["num_repetitions"]
                rep_list = [simulate_text_transmission(config, text, snr)
                            for _ in range(reps) if self.running]

                if not self.running:
                    break

                avg = {
                    "snr": snr,
                    "ber": float(np.mean([r["ber"] for r in rep_list])),
                    "ser": float(np.mean([r["ser"] for r in rep_list])),
                    "theoretical_ber": float(np.mean([r["theoretical_ber"] for r in rep_list])),
                    "theoretical_ser": float(np.mean([r["theoretical_ser"] for r in rep_list])),
                    "cer": float(np.mean([r["text_comparison"]["correct_percentage"]
                                          for r in rep_list]))
                }

            results.append(avg)
            self.queue.put(("add_row", avg))

        if self.running:
            self.current_config = config
            self.current_results = results

            results_file = save_results_to_text(config, results, mode)
            self.queue.put(("plot_data", (config, results, mode)))

            elapsed = time.time() - start_time
            self.queue.put(("progress", 100,
                            f"Завершено за {elapsed:.1f}с", ""))
            self.queue.put(("info", f"Результаты:\n{results_file}"))

            logging.info("="*60)
            logging.info(f"СИМУЛЯЦИЯ ЗАВЕРШЕНА за {elapsed:.1f} с")
            logging.info(f"Результаты сохранены в {results_file}")
            logging.info("="*60)
        else:
            self.queue.put(("progress", 0, "Остановлено", ""))
    # ---------- КОНЕЦ ИЗМЕНЕНИЙ ----------

    def process_queue(self):
        """Обработка очереди сообщений из потока симуляции"""
        try:
            while True:
                msg = self.queue.get_nowait()

                if msg[0] == "progress":
                    self.progress_var.set(msg[1])
                    self.progress_label_var.set(msg[2])
                    self.eta_var.set(msg[3] if len(msg) > 3 else "")

                elif msg[0] == "add_row":
                    s = msg[1]
                    cer = f"{s['cer']:.2f}" if s['cer'] != 0 else "—"
                    self.tree.insert('', tk.END, values=(
                        f"{s['snr']:.1f}",
                        f"{s['ber']:.2e}",
                        f"{s['ser']:.2e}",
                        f"{s.get('theoretical_ber', 0):.2e}",
                        f"{s.get('theoretical_ser', 0):.2e}",
                        cer
                    ))
                    self.tree.see(self.tree.get_children()[-1])

                elif msg[0] == "plot_data":
                    config, results, mode = msg[1]
                    try:
                        plot_file, fig = plot_and_save_results(config, results, mode,
                                                               self.show_theo_var.get())
                        if fig is not None:
                            self.display_plot(fig)
                    except Exception as e:
                        logging.error(f"Error plotting: {e}")
                        self.queue.put(("error", f"Ошибка графика: {e}"))

                elif msg[0] == "info":
                    messagebox.showinfo("✓ Информация", msg[1])

                elif msg[0] == "error":
                    messagebox.showerror("✗ Ошибка", msg[1])

        except queue.Empty:
            pass
        except Exception as e:
            logging.error(f"Queue error: {e}")

        finally:
            self.root.after(100, self.process_queue)

    def display_plot(self, fig):
        """Отображает график в интерфейсе"""
        self.clear_plot()
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self.plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

        self._current_canvas = canvas
        self._current_fig = fig
        self.root.update_idletasks()

    def clear_plot(self):
        """Очищает область графика"""
        for w in self.plot_frame.winfo_children():
            w.destroy()
        self._current_canvas = None
        self._current_fig = None

    def clear_results(self):
        """Очищает результаты"""
        for r in self.tree.get_children():
            self.tree.delete(r)
        self.clear_plot()
        self.progress_var.set(0)
        self.progress_label_var.set("Готов")
        self.eta_var.set("")
        self.running = False
        self.current_results = None
        self.current_config = None
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def stop_simulation(self):
        """Останавливает симуляцию"""
        self.running = False
        self.progress_label_var.set("Остановка...")

    def save_results(self):
        """Сохранение текущих результатов"""
        if self.current_results is None:
            messagebox.showwarning("Внимание", "Нет результатов для сохранения")
            return

        dialog = tk.Toplevel(self.root)
        dialog.title("Сохранить результаты")
        dialog.geometry("350x150")
        dialog.configure(bg=DARK_BG)

        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        ttk.Label(frame, text="Имя конфигурации:", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 5))

        name_var = tk.StringVar()
        entry = ttk.Entry(frame, textvariable=name_var, width=40)
        entry.pack(fill=tk.X, pady=(0, 15))
        entry.focus()

        def save():
            name = name_var.get().strip() or None
            result_id = results_manager.save_results(
                self.current_config,
                self.current_results,
                self.mode_var.get(),
                name
            )
            messagebox.showinfo("✓ Сохранено", f"ID: {result_id}")
            dialog.destroy()

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)
        ttk.Button(btn_frame, text="✓ Сохранить", command=save).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="✗ Отмена", command=dialog.destroy).pack(side=tk.LEFT, padx=5)

    def show_history(self):
        """Показывает историю симуляций"""
        history_window = tk.Toplevel(self.root)
        history_window.title("История симуляций")
        history_window.geometry("700x500")
        history_window.configure(bg=DARK_BG)

        frame = ttk.Frame(history_window)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(frame, text="📚 Сохраненные результаты", style='Title.TLabel',
                  foreground=ACCENT_CYAN).pack(anchor=tk.W, pady=(0, 10))

        results_list = results_manager.get_results_list()

        if not results_list:
            ttk.Label(frame, text="История пуста", foreground=ACCENT_YELLOW).pack(pady=20)
            return

        # Таблица истории
        tree_frame = ttk.Frame(frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cols = ('name', 'modulation', 'coding', 'mode', 'points', 'snr_range')
        tree = ttk.Treeview(tree_frame, columns=cols, show='headings', height=10)

        tree.heading('name', text='Имя')
        tree.heading('modulation', text='Модуляция')
        tree.heading('coding', text='Кодирование')
        tree.heading('mode', text='Режим')
        tree.heading('points', text='Точек')
        tree.heading('snr_range', text='SNR (дБ)')

        for col, width in zip(cols, [150, 100, 100, 60, 60, 120]):
            tree.column(col, width=width, anchor=tk.CENTER)

        for rid, info in [(k, v) for k, v in results_manager.index.items()]:
            snr_str = f"{info['snr_range'][0]:.0f}-{info['snr_range'][1]:.0f}"
            tree.insert('', tk.END, iid=rid, values=(
                info['name'],
                info['modulation'],
                info['coding'],
                info['mode'],
                info['num_points'],
                snr_str
            ))

        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # Кнопки управления
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X)

        def load_selected():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("Выбор", "Выберите результат")
                return

            result_id = selected[0]
            config, results = results_manager.load_results(result_id)

            self.current_config = config
            self.current_results = results

            # Очищаем таблицу и график
            for r in self.tree.get_children():
                self.tree.delete(r)
            self.clear_plot()

            # Заполняем таблицу
            for r in results:
                cer = f"{r['cer']:.2f}" if r.get('cer', 0) != 0 else "—"
                self.tree.insert('', tk.END, values=(
                    f"{r['snr']:.1f}",
                    f"{r['ber']:.2e}",
                    f"{r['ser']:.2e}",
                    f"{r.get('theoretical_ber', 0):.2e}",
                    f"{r.get('theoretical_ser', 0):.2e}",
                    cer
                ))

            # Строим график
            plot_file, fig = plot_and_save_results(config, results, config['simulation_mode'],
                                                   self.show_theo_var.get())
            if fig:
                self.display_plot(fig)

            messagebox.showinfo("✓", f"Загружены: {results_manager.index[result_id]['name']}")
            history_window.destroy()

        def delete_selected():
            selected = tree.selection()
            if not selected:
                messagebox.showwarning("Выбор", "Выберите результат")
                return

            result_id = selected[0]
            if messagebox.askyesno("Подтверждение", "Удалить результаты?"):
                results_manager.delete_results(result_id)
                tree.delete(result_id)
                messagebox.showinfo("✓", "Удалено")

        def compare_selected():
            selected = tree.selection()
            if len(selected) < 2:
                messagebox.showwarning("Выбор", "Выберите минимум 2 результата")
                return

            comparison = results_manager.compare_results(list(selected))

            # Окно сравнения
            comp_window = tk.Toplevel(history_window)
            comp_window.title("Сравнение результатов")
            comp_window.geometry("800x500")
            comp_window.configure(bg=DARK_BG)

            text = scrolledtext.ScrolledText(comp_window, bg='#1a1a2e',
                                             fg=TEXT_COLOR, font=('Courier', 9))
            text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

            # Форматированный вывод сравнения
            output = "=" * 100 + "\n"
            output += "СРАВНЕНИЕ РЕЗУЛЬТАТОВ\n"
            output += "=" * 100 + "\n\n"

            for i, res in enumerate(comparison['results']):
                output += f"▶ {res['name']}\n"
                output += f"  Модуляция: {res['config']['modulation']}\n"
                output += f"  Кодирование: {res['config']['coding']}\n"
                output += f"  {'SNR(дБ)':<10} {'BER':<12} {'SER':<12}\n"
                output += "-" * 50 + "\n"
                for d in res['data']:
                    output += f"  {d['snr']:<10.1f} {d['ber']:<12.2e} {d['ser']:<12.2e}\n"
                output += "\n"

            text.insert(tk.END, output)
            text.config(state=tk.DISABLED)

        ttk.Button(btn_frame, text="📂 Загрузить", command=load_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="🗑 Удалить", command=delete_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="📊 Сравнить", command=compare_selected).pack(side=tk.LEFT, padx=5)

    def export_results(self):
        """Экспорт результатов"""
        if self.current_results is None:
            messagebox.showwarning("Внимание", "Нет результатов для экспорта")
            return

        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV файлы", "*.csv"), ("JSON файлы", "*.json"), ("PNG графики", "*.png")]
        )

        if not filename:
            return

        try:
            if filename.endswith('.csv'):
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    if self.current_results:
                        fieldnames = self.current_results[0].keys()
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(self.current_results)
            elif filename.endswith('.json'):
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump({
                        'config': self.current_config,
                        'results': self.current_results
                    }, f, indent=2, ensure_ascii=False)
            elif filename.endswith('.png'):
                if hasattr(self, '_current_fig') and self._current_fig:
                    self._current_fig.savefig(filename, dpi=150, facecolor=DARK_BG)

            messagebox.showinfo("✓ Экспортировано", f"Файл: {filename}")
        except Exception as e:
            messagebox.showerror("✗ Ошибка", f"Ошибка экспорта: {e}")

    def show_log(self):
        """Показывает логи симуляции"""
        try:
            # Показываем последний лог-файл (самый новый в папке logs)
            log_dir = "logs"
            if os.path.exists(log_dir):
                log_files = sorted(Path(log_dir).glob("sim_*.log"), key=os.path.getmtime, reverse=True)
                if log_files:
                    lf = log_files[0]
                else:
                    # Если нет отдельных логов, показываем simulation.log (если есть)
                    lf = os.path.join(log_dir, "simulation.log")
                    if not os.path.exists(lf):
                        messagebox.showinfo("ℹ Информация", "Логи еще не созданы")
                        return
            else:
                messagebox.showinfo("ℹ Информация", "Логи еще не созданы")
                return

            if os.path.exists(lf):
                log_window = tk.Toplevel(self.root)
                log_window.title("Логи симуляции")
                log_window.geometry("700x500")
                log_window.configure(bg=DARK_BG)

                text = scrolledtext.ScrolledText(log_window, bg='#1a1a2e',
                                                 fg=ACCENT_GREEN, font=('Courier', 9),
                                                 wrap=tk.WORD)
                text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

                with open(lf, 'r', encoding='utf-8') as f:
                    text.insert(tk.END, f.read())

                text.config(state=tk.DISABLED)
            else:
                messagebox.showinfo("ℹ Информация", "Логи еще не созданы")
        except Exception as e:
            messagebox.showerror("✗ Ошибка", f"Ошибка при чтении логов: {e}")


def main():
    root = tk.Tk()
    try:
        app = SimulationGUI(root)
        root.mainloop()
    except Exception as e:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()