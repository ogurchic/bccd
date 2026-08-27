#!/usr/bin/env python3
"""
Единый скрипт для запуска всего пайплайна обучения.
Оптимизирован для работы на CPU (без видеокарты).

Использование:
    python run_pipeline.py              # быстрый тест (5 эпох)
    python run_pipeline.py --full       # полное обучение
    python run_pipeline.py --test-file audio.wav  # предсказание после обучения
"""
import sys
import argparse
import time
import torch
from pathlib import Path

# Добавляем код в PYTHONPATH
PROJECT_ROOT = Path(__file__).parent
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

# Цветной вывод для терминала
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.ENDC}\n")


def print_step(text):
    print(f"\n{Colors.CYAN}→ {text}{Colors.ENDC}")


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.ENDC}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.ENDC}")


def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.ENDC}")


# =========================================================================
# Проверки окружения
# =========================================================================

def check_environment():
    """Проверяет зависимости и устройство."""
    print_header("ПРОВЕРКА ОКРУЖЕНИЯ")

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    print(f"Python: {py_version}")

    # PyTorch version
    print(f"PyTorch: {torch.__version__}")

    # Device
    if torch.cuda.is_available():
        print(f"Устройство: {Colors.GREEN}CUDA ({torch.cuda.get_device_name(0)}){Colors.ENDC}")
        device = torch.device("cuda")
    else:
        print(f"Устройство: {Colors.YELLOW}CPU (CUDA недоступна){Colors.ENDC}")
        device = torch.device("cpu")

        # Оптимизация для CPU
        cpu_count = torch.get_num_threads()
        print(f"Доступно потоков: {cpu_count}")

        # Для Ryzen 5 3500U (4 ядра / 8 потоков) используем 4 потока
        optimal_threads = min(cpu_count, 4)
        torch.set_num_threads(optimal_threads)
        print(f"Установлено потоков: {optimal_threads}")

    # Проверка основных модулей
    try:
        import librosa
        import numpy
        import pandas
        import sklearn
        import matplotlib
        print_success("Все зависимости установлены")
    except ImportError as e:
        print_error(f"Отсутствует зависимость: {e}")
        print("Установите зависимости: pip install -r requirements-cpu.txt")
        sys.exit(1)

    return device


def check_data(raw_dir, processed_dir):
    """Проверяет наличие данных."""
    print_header("ПРОВЕРКА ДАННЫХ")

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    if not raw_path.exists():
        print_error(f"Директория с сырыми данными не найдена: {raw_path}")
        print(f"Создайте структуру: {raw_path}/cat/, {raw_path}/dog/")
        return False

    # Ищем подпапки с классами
    class_dirs = [d for d in raw_path.iterdir()
                  if d.is_dir() and not d.name.startswith('.')]

    if not class_dirs:
        print_error(f"В {raw_path} не найдено ни одной подпапки с классами")
        print(f"Создайте подпапки: {raw_path}/cat/, {raw_path}/dog/")
        return False

    print(f"Найдено классов: {len(class_dirs)}")
    total_files = 0
    for class_dir in class_dirs:
        wav_files = list(class_dir.glob("*.wav")) + list(class_dir.glob("*.WAV"))
        total_files += len(wav_files)
        print(f"  {class_dir.name}: {len(wav_files)} файлов")

    if total_files == 0:
        print_error("Не найдено .wav файлов")
        return False

    print_success(f"Всего файлов: {total_files}")

    # Проверка обработанных данных
    if processed_path.exists():
        metadata_file = processed_path / "metadata.csv"
        if metadata_file.exists():
            print_warning(f"Найдены старые обработанные данные в {processed_path}")
            print_warning("Они будут перезаписаны при запуске предобработки")

    return True


def check_checkpoint(checkpoint_dir):
    """Проверяет наличие обученной модели."""
    checkpoint_path = Path(checkpoint_dir)
    best_model = checkpoint_path / "best_model.pt"

    if best_model.exists():
        print_success(f"Найдена обученная модель: {best_model}")
        return True
    else:
        print_warning(f"Модель не найдена: {best_model}")
        return False


# =========================================================================
# Шаги пайплайна
# =========================================================================

def step_preprocessing(cfg):
    """Шаг 1: Предобработка данных."""
    print_step("Шаг 1: Предобработка данных")
    start = time.time()

    from preprocessing import process_dataset, visualize_spec

    df, class_names = process_dataset(
        raw_dir=cfg.paths.raw_dir,
        processed_dir=cfg.paths.processed_dir,
    )

    elapsed = time.time() - start
    print_success(f"Предобработка завершена за {elapsed:.1f} сек")

    # Визуализация (опционально, сохраняем без show)
    try:
        import matplotlib
        matplotlib.use('Agg')  # Backend без GUI
        visualize_spec(processed_dir=cfg.paths.processed_dir, num_samples=3)
        print_success("Визуализация спектрограмм сохранена")
    except Exception as e:
        print_warning(f"Не удалось создать визуализацию: {e}")

    return True


def step_split(cfg):
    """Шаг 2: Разбиение данных."""
    print_step("Шаг 2: Разбиение данных на train/val/test")
    start = time.time()

    from split_data import split_dataset

    train_df, val_df, test_df = split_dataset(
        processed_dir=cfg.paths.processed_dir,
    )

    elapsed = time.time() - start
    print_success(f"Разбиение завершено за {elapsed:.1f} сек")

    return True


def step_train(cfg, quick_test=False):
    """Шаг 3: Обучение модели."""
    print_step("Шаг 3: Обучение модели")
    start = time.time()

    from dataset import create_dataloaders
    from train import AudioClassifier, train_model

    # Оптимизация параметров для CPU
    if quick_test:
        print_warning("РЕЖИМ БЫСТРОГО ТЕСТА: уменьшенные параметры")
        batch_size = 8          # меньше батч для быстрой итерации
        num_epochs = 10          # мало эпох для теста
        use_augmentation = False  # отключаем аугментацию для скорости
        use_mixup = False       # отключаем mixup для скорости
    else:
        batch_size = cfg.training.batch_size
        num_epochs = cfg.training.num_epochs
        use_augmentation = cfg.training.use_augmentation
        use_mixup = cfg.training.use_mixup

    # Переопределяем конфиг для текущего запуска
    cfg.training.batch_size = batch_size
    cfg.training.num_epochs = num_epochs
    cfg.training.use_augmentation = use_augmentation
    cfg.training.use_mixup = use_mixup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Обучение на: {device}")

    # Загрузка данных
    loaders = create_dataloaders(
        processed_dir=cfg.paths.processed_dir,
        batch_size=batch_size,
    )

    # Создание модели
    model = AudioClassifier().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {total_params:,}")

    # Обучение
    import torch.nn as nn
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    # Scheduler только для полного обучения
    scheduler = None
    if not quick_test and cfg.training.use_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=cfg.training.scheduler_factor,
            patience=cfg.training.scheduler_patience,
            min_lr=cfg.training.min_lr,
            verbose=True,
        )

    history = train_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        num_epochs=num_epochs,
        save_dir=cfg.paths.checkpoint_dir,
    )

    elapsed = time.time() - start
    print_success(f"Обучение завершено за {elapsed:.1f} сек")

    return True


def step_evaluate(cfg):
    """Шаг 4: Оценка модели."""
    print_step("Шаг 4: Оценка модели на тестовых данных")
    start = time.time()

    from evaluate import evaluate_model

    model_path = Path(cfg.paths.checkpoint_dir) / "best_model.pt"
    if not model_path.exists():
        print_error("Модель для оценки не найдена")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(str(model_path), device=device)

    elapsed = time.time() - start
    print_success(f"Оценка завершена за {elapsed:.1f} сек")

    return True


def step_predict(cfg, audio_path):
    """Шаг 5: Предсказание на аудиофайле (.wav)."""
    print_step(f"Шаг 5: Предсказание на {audio_path}")
    start = time.time()

    from predict import load_model, preprocess_audio, load_class_names
    import numpy as np

    model_path = Path(cfg.paths.checkpoint_dir) / "best_model.pt"
    if not model_path.exists():
        print_error("Модель для предсказания не найдена")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загрузка модели и классов
    model = load_model(str(model_path), device)
    class_names = load_class_names(cfg.paths.processed_dir)

    # Предсказание
    spectrogram = preprocess_audio(audio_path)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = np.expand_dims(spectrogram, axis=0)
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(spectrogram)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()

    probs = probabilities[0].cpu().numpy()
    pred_class = class_names[predicted_class]
    confidence = probs[predicted_class] * 100

    print(f"\n{Colors.BOLD}Результат:{Colors.ENDC}")
    print(f"  Класс: {Colors.GREEN}{pred_class}{Colors.ENDC}")
    print(f"  Уверенность: {confidence:.1f}%")
    print(f"\n  Вероятности:")
    for i, cls in enumerate(class_names):
        prob = probs[i] * 100
        bar = '█' * int(prob / 5)
        print(f"    {cls:15}: {prob:5.1f}% {bar}")

    elapsed = time.time() - start
    print_success(f"Предсказание завершено за {elapsed:.1f} сек")

    return True

# =========================================================================
# Главный запуск
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Пайплайн обучения классификатора звуков",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python run_pipeline.py                  # быстрый тест (5 эпох)
  python run_pipeline.py --full           # полное обучение
  python run_pipeline.py --test-file cat.wav  # предсказание после обучения
        """,
    )
    parser.add_argument('--full', action='store_true',
                        help='Полное обучение (все эпохи, аугментация, mixup)')
    parser.add_argument('--test-file', type=str, default=None,
                        help='Путь к аудиофайлу для предсказания после обучения')
    parser.add_argument('--raw-dir', type=str, default='data/raw',
                        help='Директория с сырыми данными')
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                        help='Директория для обработанных данных')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Директория для чекпоинтов')

    args = parser.parse_args()

    # Загружаем конфиг после парсинга аргументов
    from config import cfg

    # Переопределяем пути из аргументов
    cfg.paths.raw_dir = args.raw_dir
    cfg.paths.processed_dir = args.processed_dir
    cfg.paths.checkpoint_dir = args.checkpoint_dir

    print_header("BCCD ПАЙПЛАЙН")
    print(f"Режим: {'ПОЛНОЕ ОБУЧЕНИЕ' if args.full else 'БЫСТРЫЙ ТЕСТ'}")
    print(f"Данные: {cfg.paths.raw_dir}")
    print(f"Чекпоинты: {cfg.paths.checkpoint_dir}")

    # Шаг 1: Проверка окружения
    device = check_environment()

    # Шаг 2: Проверка данных
    if not check_data(cfg.paths.raw_dir, cfg.paths.processed_dir):
        sys.exit(1)

    # Шаг 3: Пайплайн
    total_start = time.time()

    try:
        step_preprocessing(cfg)
        step_split(cfg)
        step_train(cfg, quick_test=not args.full)
        step_evaluate(cfg)

                # Предсказание, если указан файл
        if args.test_file:
            test_file = Path(args.test_file)
            if test_file.exists():
                step_predict(cfg, test_file)
            else:
                print_warning(f"Файл для предсказания не найден: {test_file}")
        else:
            # Ищем первый .wav файл из теста
            test_dir = Path(cfg.paths.processed_dir)
            test_csv = test_dir / "test.csv"
            if test_csv.exists():
                import pandas as pd
                test_df = pd.read_csv(test_csv)
                if len(test_df) > 0 and 'source_path' in test_df.columns:
                    # Берем путь к исходному .wav файлу
                    first_test_path = test_df.iloc[0]['source_path']
                    if Path(first_test_path).exists():
                        print_warning(f"Файл для предсказания не указан. "
                                    f"Использую первый из теста: {first_test_path}")
                        step_predict(cfg, Path(first_test_path))
                    else:
                        print_warning(f"Исходный файл не найден: {first_test_path}")
                else:
                    print_warning("В test.csv нет колонки 'source_path' или файл пуст")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Прервано пользователем{Colors.ENDC}")
        sys.exit(1)
    except Exception as e:
        print_error(f"Ошибка в пайплайне: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    total_elapsed = time.time() - total_start
    print_header("ПАЙПЛАЙН ЗАВЕРШЁН")
    print(f"Общее время: {total_elapsed:.1f} сек ({total_elapsed/60:.1f} мин)")
    print(f"Чекпоинты: {cfg.paths.checkpoint_dir}/")
    print(f"Модель: {cfg.paths.checkpoint_dir}/best_model.pt")
    print(f"История: {cfg.paths.checkpoint_dir}/history.csv")


if __name__ == "__main__":
    main()