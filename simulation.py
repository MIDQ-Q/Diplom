import os
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
import json

# Импортируем классы модуляции и кодирования
from modulation import PSKModulator, QAMModulator, theoretical_ber_psk, theoretical_ser_psk, theoretical_ber_qam, \
    theoretical_ser_qam
from coding import HammingCoder, LDPCCoder
from results_manager import ResultsManager
from channel import CompositeChannelModel

# --- Настройка логирования ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "simulation.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='a'
)

# Инициализируем менеджер результатов
results_manager = ResultsManager()


# ---------- Helper конверсии ----------
def text_to_bits(text: str, encoding: str = 'utf-8') -> np.ndarray:
    """Конвертирует текст в биты"""
    ba = text.encode(encoding)
    bits = []
    for byte in ba:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return np.array(bits, dtype=int)


def bits_to_text(bits: np.ndarray, encoding: str = 'utf-8') -> str:
    """Конвертирует биты в текст"""
    b = np.array(bits, dtype=int).flatten()
    pad = (8 - (len(b) % 8)) % 8
    if pad:
        b = np.concatenate([b, np.zeros(pad, dtype=int)])
    out = bytearray()
    for i in range(0, len(b), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(b[i + j])
        out.append(byte)
    try:
        return out.decode(encoding)
    except Exception:
        return out.decode(encoding, errors='replace')


def compare_texts(original: str, received: str) -> Dict:
    """Сравнивает оригинальный и восстановленный текст"""
    total = len(original)
    compared = min(len(original), len(received))
    correct = sum(1 for i in range(compared) if original[i] == received[i])
    correct_percentage = (correct / compared * 100) if compared > 0 else 0.0
    return {
        "total_original_chars": total,
        "total_received_chars": len(received),
        "compared_chars": compared,
        "correct_chars": correct,
        "correct_percentage": correct_percentage
    }


# ---------- Фабричные функции ----------
def create_modulator(config):
    """Создает модулятор по конфигу"""
    mod_type = config["modulation"]["type"]
    order = config["modulation"]["order"]
    use_gray = config["modulation"]["use_gray_code"]

    if mod_type == "PSK":
        return PSKModulator(M=order, use_gray_code=use_gray)
    elif mod_type == "QAM":
        return QAMModulator(M=order, use_gray_code=use_gray)
    else:
        raise ValueError(f"Неизвестный тип модуляции: {mod_type}")


def create_coder(config):
    """Создает кодер по конфигу"""
    if not config["coding"]["enabled"]:
        return None, 1.0

    coding_type = config["coding"]["type"]
    n = config["coding"]["n"]
    k = config["coding"]["k"]

    if coding_type == "hamming":
        coder = HammingCoder(n=n, k=k)
        code_rate = k / n
    elif coding_type == "ldpc":
        coder = LDPCCoder(n=n, k=k)
        code_rate = k / n
    else:
        raise ValueError(f"Неизвестный тип кодирования: {coding_type}")

    return coder, code_rate


def theoretical_ber(config, ebn0_dB):
    """Вычисляет теоретический BER"""
    mod_type = config["modulation"]["type"]
    order = config["modulation"]["order"]
    use_gray = config["modulation"]["use_gray_code"]

    if mod_type == "PSK":
        return theoretical_ber_psk(ebn0_dB, order, use_gray)
    elif mod_type == "QAM":
        return theoretical_ber_qam(ebn0_dB, order, use_gray)
    else:
        return 0.0


def theoretical_ser(config, ebn0_dB):
    """Вычисляет теоретический SER"""
    mod_type = config["modulation"]["type"]
    order = config["modulation"]["order"]
    use_gray = config["modulation"]["use_gray_code"]

    if mod_type == "PSK":
        return theoretical_ser_psk(ebn0_dB, order, use_gray)
    elif mod_type == "QAM":
        return theoretical_ser_qam(ebn0_dB, order, use_gray)
    else:
        return 0.0


def log_config(config, mode, data_bits_len=None, text_len=None):
    """Логирует полную конфигурацию симуляции (один раз в начале)"""
    lines = []
    lines.append("=" * 80)
    lines.append("НОВАЯ СИМУЛЯЦИЯ")
    lines.append(f"Режим: {mode}")
    lines.append(f"Модуляция: {config['modulation']['type']}-{config['modulation']['order']}, Код Грея: {config['modulation']['use_gray_code']}")
    if config['coding']['enabled']:
        lines.append(f"Кодирование: {config['coding']['type']} ({config['coding']['n']},{config['coding']['k']}), скорость={config['coding']['k']/config['coding']['n']:.3f}")
    else:
        lines.append("Кодирование: отключено")
    # Параметры канала
    ch = config['channel']
    active_ch = []
    if ch['rayleigh']['enabled']:
        active_ch.append(f"Rayleigh(лучей={ch['rayleigh']['num_rays']}, доплер={ch['rayleigh']['normalized_doppler']})")
    if ch['rician']['enabled']:
        active_ch.append(f"Rician(лучей={ch['rician']['num_rays']}, доплер={ch['rician']['normalized_doppler']}, K={ch['rician']['rician_factor_k']})")
    if ch['phase_noise']['enabled']:
        active_ch.append(f"PhaseNoise(σ²={ch['phase_noise']['phase_noise_variance']})")
    if ch['frequency_offset']['enabled']:
        active_ch.append(f"FreqOffset(δf={ch['frequency_offset']['normalized_freq_offset']})")
    if ch['timing_offset']['enabled']:
        active_ch.append(f"TimingOffset(range={ch['timing_offset']['timing_offset_range']})")
    if ch['impulse_noise']['enabled']:
        active_ch.append(f"ImpulseNoise(p={ch['impulse_noise']['impulse_probability']}, A={ch['impulse_noise']['impulse_amplitude_sigma']}σ, ширина={ch['impulse_noise']['impulse_width_from']}-{ch['impulse_noise']['impulse_width_to']})")
    if not active_ch:
        active_ch.append("AWGN")
    lines.append(f"Каналы: {', '.join(active_ch)}")
    if data_bits_len is not None:
        lines.append(f"Длина данных: {data_bits_len} бит")
    if text_len is not None:
        lines.append(f"Длина текста: {text_len} символов")
    lines.append(f"Диапазон Eb/N0: {min(config['ebn0_dB_range'])}..{max(config['ebn0_dB_range'])} дБ, шаг {config['ebn0_dB_range'][1]-config['ebn0_dB_range'][0] if len(config['ebn0_dB_range'])>1 else 'N/A'}")
    lines.append("=" * 80)
    for line in lines:
        logging.info(line)


# ---------- Основные функции симуляции ----------
def simulate_transmission(config: Dict, ebn0_dB: float,
                          data_bits: Optional[np.ndarray] = None) -> Dict:
    """
    Симуляция передачи при фиксированном SNR с поддержкой когерентного приёма.
    """
    mod = create_modulator(config)
    coder, code_rate = create_coder(config)
    channel = CompositeChannelModel(config.get('channel', {}))

    if data_bits is None:
        num_bits = config["random_settings"]["num_bits"]
        data_bits = np.random.randint(0, 2, num_bits, dtype=int)

    if not hasattr(simulate_transmission, "_config_logged"):
        log_config(config, "random", data_bits_len=len(data_bits))
        simulate_transmission._config_logged = True

    if coder is not None:
        tx_bits = coder.encode(data_bits)
    else:
        tx_bits = data_bits.copy()

    tx_symbols = mod.modulate(tx_bits)

    bits_per_symbol = mod.bits_per_symbol
    ebn0_linear = 10 ** (ebn0_dB / 10.0)
    snr_linear = ebn0_linear * code_rate * bits_per_symbol

    # Применяем канал, получаем символы и коэффициенты
    rx_symbols, channel_coeff = channel.apply_with_coeff(tx_symbols, snr_linear)

    # Демодуляция с коррекцией
    rx_bits = mod.demodulate(rx_symbols, channel_coeff)

    # Декодирование (без изменений)
    if coder is not None:
        decoded_bits, coding_stats = coder.decode(rx_bits)
    else:
        decoded_bits = rx_bits
        coding_stats = {
            "corrected_errors": 0,
            "detected_errors": 0,
            "total_blocks": 0,
            "error_positions": []
        }

    decoded_bits = decoded_bits[:len(data_bits)]

    min_len = min(len(data_bits), len(decoded_bits))
    bit_errors = np.sum(data_bits[:min_len] != decoded_bits[:min_len])
    ber = bit_errors / min_len if min_len > 0 else 0.0

    # SER (оставляем как было)
    if mod.M == 2 or (hasattr(mod, 'M') and mod.M == 4 and isinstance(mod, QAMModulator)):
        ser = 1 - (1 - ber) ** mod.bits_per_symbol if ber < 1 else 1.0
    else:
        sym_err = 0
        total_symbols = min(len(tx_bits), len(rx_bits)) // mod.bits_per_symbol
        for i in range(total_symbols):
            a = tx_bits[i * mod.bits_per_symbol:(i + 1) * mod.bits_per_symbol]
            b = rx_bits[i * mod.bits_per_symbol:(i + 1) * mod.bits_per_symbol]
            if not np.array_equal(a, b):
                sym_err += 1
        ser = sym_err / total_symbols if total_symbols > 0 else 0.0

    theoretical_ber_val = theoretical_ber(config, ebn0_dB)
    theoretical_ser_val = theoretical_ser(config, ebn0_dB)

    channel_names = channel.get_channel_names()
    esn0_dB = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

    log_msg = (
        f"Eb/N0={ebn0_dB:.2f} dB | Es/N0={esn0_dB:.2f} dB | "
        f"Mod={mod.__class__.__name__}-{mod.M} | Gray={mod.use_gray_code} | "
        f"CodeRate={code_rate:.3f} | Bits={min_len} | BitErrors={bit_errors} | "
        f"BER={ber:.3e} | SER={ser:.3e} | "
        f"CorrectedErrors={coding_stats.get('corrected_errors', 0)} | "
        f"DetectedErrors={coding_stats.get('detected_errors', 0)} | "
        f"Blocks={coding_stats.get('total_blocks', 0)} | "
        f"Channels={channel_names}"
    )
    logging.info(log_msg)

    return {
        "snr": ebn0_dB,
        "ber": float(ber),
        "ser": float(ser),
        "theoretical_ber": float(theoretical_ber_val),
        "theoretical_ser": float(theoretical_ser_val),
        "Es": 1.0,
        "Eb": 1.0 / (mod.bits_per_symbol * code_rate),
        "spectral_efficiency": float(mod.bits_per_symbol * code_rate),
        "corrected_errors": coding_stats.get("corrected_errors", 0),
        "detected_errors": coding_stats.get("detected_errors", 0),
        "total_blocks": coding_stats.get("total_blocks", 0),
        "active_channels": channel_names
    }


def simulate_text_transmission(config: Dict, text: str, ebn0_dB: float) -> Dict:
    """
    Симуляция передачи текста при фиксированном SNR.
    """
    mod = create_modulator(config)
    coder, code_rate = create_coder(config)
    channel = CompositeChannelModel(config.get('channel', {}))

    # Логируем конфигурацию один раз
    if not hasattr(simulate_text_transmission, "_config_logged"):
        log_config(config, "text", text_len=len(text))
        simulate_text_transmission._config_logged = True

    original_bits = text_to_bits(text, config["text_settings"]["text_encoding"])
    max_bits = config["text_settings"]["max_text_length"] * 8
    if len(original_bits) > max_bits:
        original_bits = original_bits[:max_bits]

    if coder is not None:
        tx_bits = coder.encode(original_bits)
    else:
        tx_bits = original_bits.copy()

    tx_symbols = mod.modulate(tx_bits)

    bits_per_symbol = mod.bits_per_symbol
    ebn0_linear = 10 ** (ebn0_dB / 10.0)
    snr_linear = ebn0_linear * code_rate * bits_per_symbol

    rx_symbols = channel.apply(tx_symbols, snr_linear)
    rx_bits = mod.demodulate(rx_symbols)

    if coder is not None:
        decoded_bits, coding_stats = coder.decode(rx_bits)
    else:
        decoded_bits = rx_bits
        coding_stats = {
            "corrected_errors": 0,
            "detected_errors": 0,
            "total_blocks": 0,
            "error_positions": []
        }

    decoded_bits = decoded_bits[:len(original_bits)]
    decoded_text = bits_to_text(decoded_bits, config["text_settings"]["text_encoding"])

    min_len = min(len(original_bits), len(decoded_bits))
    bit_errors = np.sum(original_bits[:min_len] != decoded_bits[:min_len])
    ber = bit_errors / min_len if min_len > 0 else 0.0

    if mod.M == 2 or (hasattr(mod, 'M') and mod.M == 4 and isinstance(mod, QAMModulator)):
        ser = 1 - (1 - ber) ** mod.bits_per_symbol if ber < 1 else 1.0
    else:
        total_symbols = min(len(tx_bits), len(rx_bits)) // mod.bits_per_symbol
        sym_err = 0
        for i in range(total_symbols):
            if not np.array_equal(
                    tx_bits[i * mod.bits_per_symbol:(i + 1) * mod.bits_per_symbol],
                    rx_bits[i * mod.bits_per_symbol:(i + 1) * mod.bits_per_symbol]
            ):
                sym_err += 1
        ser = sym_err / total_symbols if total_symbols > 0 else 0.0

    theoretical_ber_val = theoretical_ber(config, ebn0_dB)
    theoretical_ser_val = theoretical_ser(config, ebn0_dB)
    text_comparison = compare_texts(text, decoded_text)

    channel_names = channel.get_channel_names()
    esn0_dB = 10 * np.log10(snr_linear) if snr_linear > 0 else -np.inf

    log_msg = (
        f"TEXT Eb/N0={ebn0_dB:.2f} dB | Es/N0={esn0_dB:.2f} dB | "
        f"Mod={mod.__class__.__name__}-{mod.M} | Gray={mod.use_gray_code} | "
        f"CodeRate={code_rate:.3f} | Bits={min_len} | BitErrors={bit_errors} | "
        f"BER={ber:.3e} | SER={ser:.3e} | CER={text_comparison['correct_percentage']:.2f}% | "
        f"CorrectedErrors={coding_stats.get('corrected_errors', 0)} | "
        f"DetectedErrors={coding_stats.get('detected_errors', 0)} | "
        f"Blocks={coding_stats.get('total_blocks', 0)} | "
        f"Channels={channel_names}"
    )
    logging.info(log_msg)

    return {
        "snr": ebn0_dB,
        "text": decoded_text,
        "ber": float(ber),
        "ser": float(ser),
        "theoretical_ber": float(theoretical_ber_val),
        "theoretical_ser": float(theoretical_ser_val),
        "original_text": text,
        "text_comparison": text_comparison,
        "active_channels": channel_names
    }


# ---------- Сохранение и графики ----------
def save_results_to_text(config: Dict, results: List[Dict], mode: str,
                         execution_time: float = None) -> str:
    """Сохраняет результаты в текстовый файл"""
    mod_type = config["modulation"]["type"]
    order = config["modulation"]["order"]
    mode_abbr = "Text" if mode == "text" else "Random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{mod_type}{order}_{mode_abbr}_{timestamp}.txt"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("РЕЗУЛЬТАТЫ СИМУЛЯЦИИ ЦИФРОВОЙ СВЯЗИ\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Режим:                 {mode}\n")
        f.write(f"Модуляция:             {mod_type}-{order}\n")
        f.write(
            f"Кодирование:           {config['coding']['type'].upper() if config['coding']['enabled'] else 'ОТСУТСТВУЕТ'}\n")
        if config['coding']['enabled']:
            f.write(f"  Тип кода:            ({config['coding']['n']},{config['coding']['k']})\n")
            f.write(f"  Скорость кода:       {config['coding']['k']}/{config['coding']['n']}\n")
        f.write(f"Код Грея:              {'ДА' if config['modulation']['use_gray_code'] else 'НЕТ'}\n")
        f.write(f"Eb/N0 диапазон (дБ):   {min(config['ebn0_dB_range']):.1f} - {max(config['ebn0_dB_range']):.1f}\n")

        # Информация о канале
        f.write("\nМодели канала:\n")
        active_channels = set()
        for r in results:
            for ch in r.get('active_channels', []):
                active_channels.add(ch)
        if active_channels:
            for ch in active_channels:
                f.write(f"  • {ch}\n")
        else:
            f.write("  • AWGN\n")

        if execution_time:
            f.write(f"\nВремя выполнения:      {execution_time:.2f} сек\n")
        f.write("\n" + "=" * 80 + "\n\n")

        f.write(f"{'SNR(дБ)':<10} {'BER':<12} {'SER':<12} {'Теор.BER':<12} {'Теор.SER':<12}")
        if mode == "text":
            f.write(f" {'CER(%)':<10}\n")
        else:
            f.write("\n")
        f.write("-" * 80 + "\n")

        for r in results:
            cer = r.get('cer', 0) if 'cer' in r else \
                (r.get('text_comparison', {}).get('correct_percentage', 0) if mode == 'text' else 0)
            f.write(
                f"{r['snr']:<10.1f} {r['ber']:<12.2e} {r['ser']:<12.2e} {r.get('theoretical_ber', 0):<12.2e} {r.get('theoretical_ser', 0):<12.2e}")
            if mode == "text":
                f.write(f" {cer:<10.2f}\n")
            else:
                f.write("\n")

        f.write("\n" + "=" * 80 + "\n")

    # Логируем завершение
    logging.info(f"Результаты сохранены в файл: {filename}")
    return filename


def plot_and_save_results(config: Dict, results: List[Dict], mode: str,
                          show_theoretical: bool = True) -> Tuple[str, plt.Figure]:
    """Строит и сохраняет графики результатов"""
    if not results:
        return None, None

    mod_type = config["modulation"]["type"]
    order = config["modulation"]["order"]
    mode_abbr = "Text" if mode == "text" else "Random"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"Plot_{mod_type}{order}_{mode_abbr}_{timestamp}.png"

    snr = np.array([r['snr'] for r in results])
    ber = np.array([r['ber'] for r in results])
    ser = np.array([r['ser'] for r in results])

    # Стиль
    plt.style.use('dark_background')

    if mode == "text":
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # График BER/SER
        ax1.semilogy(snr, ber, 'o-', color='#00d9ff', linewidth=2.5,
                     markersize=6, label='BER (эксп.)', markerfacecolor='#00d9ff',
                     markeredgewidth=1.5, markeredgecolor='white')
        ax1.semilogy(snr, ser, 's--', color='#ff006e', linewidth=2.5,
                     markersize=6, label='SER (эксп.)', markerfacecolor='#ff006e',
                     markeredgewidth=1.5, markeredgecolor='white')

        if show_theoretical:
            theo_ber = np.array([r.get('theoretical_ber', 0) for r in results])
            theo_ser = np.array([r.get('theoretical_ser', 0) for r in results])
            ax1.semilogy(snr, theo_ber, ':', color='#39ff14', linewidth=2,
                         label='Теор. BER', alpha=0.8)
            ax1.semilogy(snr, theo_ser, '-.', color='#ffbe0b', linewidth=2,
                         label='Теор. SER', alpha=0.8)

        ax1.grid(True, alpha=0.25, linestyle='--')
        ax1.set_xlabel('Eb/N0 (дБ)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Вероятность ошибки', fontsize=11, fontweight='bold')
        ax1.set_title('BER и SER', fontsize=12, fontweight='bold', pad=15)
        ax1.legend(fontsize=10, loc='upper right', framealpha=0.95)
        ax1.set_ylim(1e-6, 1)

        # График CER
        cer = np.array([r.get('cer', 0) if 'cer' in r else
                        (r.get('text_comparison', {}).get('correct_percentage', 0)
                         if 'text_comparison' in r else 0) for r in results])
        ax2.plot(snr, cer, '^-', color='#00d9ff', linewidth=2.5, markersize=8,
                 markerfacecolor='#00d9ff', markeredgewidth=1.5, markeredgecolor='white')
        ax2.grid(True, alpha=0.25, linestyle='--')
        ax2.set_xlabel('Eb/N0 (дБ)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Правильных символов (%)', fontsize=11, fontweight='bold')
        ax2.set_title('Качество восстановления текста', fontsize=12, fontweight='bold', pad=15)
        ax2.set_ylim(-2, 105)
        ax2.axhline(y=90, color='#ffbe0b', linestyle='--', alpha=0.5, linewidth=1.5, label='90%')
        ax2.axhline(y=95, color='#39ff14', linestyle='--', alpha=0.5, linewidth=1.5, label='95%')
        ax2.axhline(y=99, color='#00d9ff', linestyle='--', alpha=0.5, linewidth=1.5, label='99%')
        ax2.legend(fontsize=10, loc='lower right', framealpha=0.95)

    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.semilogy(snr, ber, 'o-', color='#00d9ff', linewidth=2.5,
                    markersize=6, label='BER (эксп.)', markerfacecolor='#00d9ff',
                    markeredgewidth=1.5, markeredgecolor='white')
        ax.semilogy(snr, ser, 's--', color='#ff006e', linewidth=2.5,
                    markersize=6, label='SER (эксп.)', markerfacecolor='#ff006e',
                    markeredgewidth=1.5, markeredgecolor='white')

        if show_theoretical:
            theo_ber = np.array([r.get('theoretical_ber', 0) for r in results])
            theo_ser = np.array([r.get('theoretical_ser', 0) for r in results])
            ax.semilogy(snr, theo_ber, ':', color='#39ff14', linewidth=2,
                        label='Теор. BER', alpha=0.8)
            ax.semilogy(snr, theo_ser, '-.', color='#ffbe0b', linewidth=2,
                        label='Теор. SER', alpha=0.8)

        ax.grid(True, alpha=0.25, linestyle='--')
        ax.set_xlabel('Eb/N0 (дБ)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Вероятность ошибки', fontsize=11, fontweight='bold')
        ax.set_title(f'{mod_type}-{order} ({mode_abbr})', fontsize=12, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
        ax.set_ylim(1e-6, 1)

    fig.patch.set_facecolor('#1a1a2e')
    fig.tight_layout()
    fig.savefig(plot_filename, dpi=150, facecolor='#1a1a2e')

    logging.info(f"График сохранён: {plot_filename}")
    return plot_filename, fig