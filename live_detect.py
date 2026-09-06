"""
Непрерывное распознавание команд с микрофона.

Логика:
1. Читаем аудио чанками по 100 мс в скользящее окно 1 сек
2. Если громкость чанка выше порога — прогоняем окно через модель
3. Если уверенность выше порога и прошёл cooldown — ТРИГГЕР
"""

import sys
import time
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "code"))

import numpy as np
import torch
import sounddevice as sd

from config import cfg
from predict import load_model, load_class_names
from preprocessing import waveform_to_melspec

# ============ НАСТРОЙКИ ============
ENERGY_THRESHOLD = 0.03      # порог громкости (пока что подбирается ручками в mic_check.py)
CONFIDENCE_THRESHOLD = 0.7   # мин. уверенность слова
COOLDOWN_SEC = 0.5           # пауза между триггерами (сек)
HOP_SEC = 0.05                # шаг проверки (сек)
LOG_FILE = "live_detect_log.txt"
# ===================================


def format_time(seconds):
    """Форматирует секунды в MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = Path(cfg.paths.checkpoint_dir) / 'best_model.pt'

    print("Загрузка модели...")
    model = load_model(str(model_path), device)
    class_names = load_class_names()

    sr = cfg.audio.sample_rate
    win = int(sr * cfg.audio.max_duration_sec)   # окно 1 сек
    chunk = int(sr * HOP_SEC)                    # чанк 100 мс

    buf = np.zeros(win, dtype=np.float32)
    last_trigger = 0.0

    # Статистика
    start_time = time.time()
    total_checks = 0
    trigger_count = 0
    class_stats = {name: 0 for name in class_names}

    # Логирование
    log_file = None
    if LOG_FILE:
        log_path = Path(LOG_FILE)
        log_file = open(log_path, 'a', encoding='utf-8')
        log_file.write(f"\n{'='*60}\n")
        log_file.write(f"Новая сессия: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"{'='*60}\n")
        log_file.flush()
        print(f"Лог сохраняется в: {log_path}")

    print(f"Устройство: {device}")
    print(f"Классы: {class_names}")
    print(f"Порог громкости: {ENERGY_THRESHOLD}, порог уверенности: {CONFIDENCE_THRESHOLD}")
    print("Слушаю... (Ctrl+C — стоп)\n")

    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype='float32',
                            blocksize=chunk) as stream:
            while True:
                data, _ = stream.read(chunk)
                x = data[:, 0]

                # Сдвигаем скользящее окно и дописываем новый чанк
                buf[:-chunk] = buf[chunk:]
                buf[-chunk:] = x

                rms = float(np.sqrt(np.mean(x ** 2)))
                now = time.time()
                total_checks += 1

                # Гейт: тишина или cooldown — не тратим время на инференс
                if rms < ENERGY_THRESHOLD:
                    continue
                if now - last_trigger < COOLDOWN_SEC:
                    continue

                # Инференс на скользящем окне
                spec = waveform_to_melspec(buf.copy())
                t = torch.tensor(spec, dtype=torch.float32)[None, None].to(device)

                with torch.no_grad():
                    probs = torch.softmax(model(t), dim=1)[0]
                idx = int(probs.argmax().item())
                conf = float(probs[idx].item())

                if conf >= CONFIDENCE_THRESHOLD:
                    class_name = class_names[idx]
                    elapsed = now - start_time
                    elapsed_str = format_time(elapsed)

                    # Обновляем статистику
                    trigger_count += 1
                    class_stats[class_name] += 1

                    # Вывод в консоль
                    print(f"🔔 [{elapsed_str}] {class_name:6s} | "
                          f"уверенность {conf:.0%} | громкость {rms:.3f} | "
                          f"срабатывание #{trigger_count}")

                    # Логирование в файл
                    if log_file:
                        log_file.write(f"[{elapsed_str}] {class_name:6s} | "
                                      f"уверенность {conf:.1%} | "
                                      f"громкость {rms:.3f}\n")
                        log_file.flush()

                    last_trigger = now

    except KeyboardInterrupt:
        print('\n\n📊 СТАТИСТИКА СЕССИИ')
        print('=' * 60)
        elapsed = time.time() - start_time
        print(f"Длительность: {format_time(elapsed)}")
        print(f"Всего проверок: {total_checks}")
        print(f"Всего срабатываний: {trigger_count}")
        print(f"Частота проверок: {total_checks / elapsed:.1f} раз/сек")
        print(f"\nРаспределение по классам:")
        for name, count in sorted(class_stats.items(), key=lambda x: -x[1]):
            pct = (count / trigger_count * 100) if trigger_count > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"  {name:10s}: {count:3d} ({pct:5.1f}%) {bar}")

        if log_file:
            log_file.write(f"\n{'='*60}\n")
            log_file.write(f"Конец сессии: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Длительность: {format_time(elapsed)}\n")
            log_file.write(f"Всего проверок: {total_checks}\n")
            log_file.write(f"Всего срабатываний: {trigger_count}\n")
            log_file.write(f"Распределение по классам:\n")
            for name, count in sorted(class_stats.items(), key=lambda x: -x[1]):
                pct = (count / trigger_count * 100) if trigger_count > 0 else 0
                log_file.write(f"  {name}: {count} ({pct:.1f}%)\n")
            log_file.write(f"{'='*60}\n")
            log_file.close()
            print(f"\nЛог сохранён в: {LOG_FILE}")


if __name__ == '__main__':
    main()