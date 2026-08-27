"""
Скачивание и подготовка датасета Speech Commands v2.
Источник: http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz

Использование:
    python code/download_speech_commands.py
    python code/download_speech_commands.py --classes 10  # только 10 команд
"""
import os
import sys
import tarfile
import urllib.request
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

# URL датасета
DATASET_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_SIZE_GB = 2.3  # примерный размер

# Все 35 классов в датасете
ALL_CLASSES = [
    'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five',
    'follow', 'forward', 'four', 'go', 'happy', 'house', 'learn', 'left',
    'marvin', 'nine', 'no', 'off', 'on', 'one', 'right', 'seven',
    'sheila', 'six', 'stop', 'three', 'tree', 'two', 'up', 'visual',
    'wow', 'yes', 'zero'
]

# Рекомендуемые 10 классов для учебного проекта
RECOMMENDED_10 = ['yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go', 'cat', 'dog']

# Минимальный набор для быстрого старта
QUICK_4 = ['yes', 'no', 'cat', 'dog']


class DownloadProgressBar(tqdm):
    """Прогресс-бар для скачивания."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_dataset(output_dir, url=DATASET_URL):
    """Скачивает архив датасета."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    archive_path = output_path / "speech_commands_v0.02.tar.gz"
    
    if archive_path.exists():
        print(f"✓ Архив уже существует: {archive_path}")
        response = input("  Удалить и скачать заново? (y/N): ")
        if response.lower() == 'y':
            archive_path.unlink()
        else:
            return archive_path
    
    print(f"\nСкачивание датасета (~{DATASET_SIZE_GB} ГБ)...")
    print(f"URL: {url}")
    print(f"Куда: {archive_path}")
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='Загрузка') as t:
        urllib.request.urlretrieve(url, filename=archive_path, reporthook=t.update_to)
    
    print(f"✓ Архив скачан: {archive_path}")
    return archive_path


def extract_dataset(archive_path, extract_dir):
    """Распаковывает архив."""
    extract_path = Path(extract_dir)
    
    # Проверяем, не распакован ли уже
    if (extract_path / "yes").exists():
        print(f"✓ Датасет уже распакован: {extract_path}")
        return extract_path
    
    print(f"\nРаспаковка архива в {extract_path}...")
    extract_path.mkdir(parents=True, exist_ok=True)
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        # Получаем список файлов для прогресс-бара
        members = tar.getmembers()
        total = len(members)
        
        with tqdm(total=total, desc='Распаковка') as pbar:
            for member in members:
                try:
                    tar.extract(member, extract_path)
                except Exception as e:
                    print(f"  ⚠ Ошибка: {member.name}: {e}")
                pbar.update(1)
    
    print(f"✓ Датасет распакован: {extract_path}")
    return extract_path


def prepare_subset(source_dir, target_dir, classes, max_files_per_class=None):
    """
    Копирует подмножество классов из полного датасета.
    
    Args:
        source_dir: папка с распакованным датасетом
        target_dir: папка для подготовленных данных (data/raw)
        classes: список классов для копирования
        max_files_per_class: макс. файлов на класс (для быстрого теста)
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    
    print(f"\nПодготовка подмножества датасета:")
    print(f"  Источник: {source_path}")
    print(f"  Назначение: {target_path}")
    print(f"  Классы ({len(classes)}): {classes}")
    if max_files_per_class:
        print(f"  Макс. файлов на класс: {max_files_per_class}")
    
    total_copied = 0
    
    for class_name in classes:
        class_source = source_path / class_name
        class_target = target_path / class_name
        
        if not class_source.exists():
            print(f"  ⚠ Класс не найден: {class_name}")
            continue
        
        class_target.mkdir(parents=True, exist_ok=True)
        
        wav_files = list(class_source.glob("*.wav"))
        if max_files_per_class:
            wav_files = wav_files[:max_files_per_class]
        
        print(f"  {class_name}: копирую {len(wav_files)} файлов...")
        
        for wav_file in tqdm(wav_files, desc=f"  {class_name}", leave=False):
            shutil.copy2(wav_file, class_target / wav_file.name)
            total_copied += 1
    
    print(f"\n✅ Скопировано {total_copied} файлов в {target_path}")
    return total_copied


def cleanup(source_dir, keep_archive=True):
    """Опционально удаляет распакованные данные для экономии места."""
    source_path = Path(source_dir)
    response = input(f"\nУдалить распакованные данные из {source_path}? (y/N): ")
    
    if response.lower() == 'y':
        shutil.rmtree(source_path)
        print(f"✓ Удалено: {source_path}")
        
        if not keep_archive:
            archive = source_path.parent / "speech_commands_v0.02.tar.gz"
            if archive.exists():
                archive.unlink()
                print(f"✓ Удалён архив: {archive}")


def main():
    parser = argparse.ArgumentParser(
        description="Скачивание датасета Speech Commands v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Примеры:
  python code/download_speech_commands.py                    # все 35 классов
  python code/download_speech_commands.py --classes 10       # 10 рекомендуемых
  python code/download_speech_commands.py --classes 4        # быстрый тест (4 класса)
  python code/download_speech_commands.py --max-files 100    # по 100 файлов на класс
  
Доступные классы ({len(ALL_CLASSES)}):
  {', '.join(ALL_CLASSES[:10])}...
        """,
    )
    parser.add_argument('--classes', type=int, default=10, choices=[4, 10, 35],
                        help='Количество классов: 4, 10 или 35 (по умолчанию 10)')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Максимум файлов на класс (для быстрого теста)')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Базовая директория для данных')
    parser.add_argument('--keep-archive', action='store_true',
                        help='Не удалять архив после распаковки')
    
    args = parser.parse_args()
    
    # Определяем классы
    if args.classes == 4:
        classes = QUICK_4
    elif args.classes == 10:
        classes = RECOMMENDED_10
    else:
        classes = ALL_CLASSES
    
    # Пути
    data_dir = Path(args.data_dir)
    download_dir = data_dir / "downloads"
    extract_dir = data_dir / "speech_commands_full"
    raw_dir = data_dir / "raw"
    
    print("=" * 60)
    print("SPEECH COMMANDS v2 - СКАЧИВАНИЕ И ПОДГОТОВКА")
    print("=" * 60)
    print(f"Классов: {args.classes}")
    print(f"Классы: {classes}")
    print(f"Макс. файлов на класс: {args.max_files or 'все'}")
    print("=" * 60)
    
    # Шаг 1: Скачивание
    archive_path = download_dataset(download_dir)
    
    # Шаг 2: Распаковка
    extract_dataset(archive_path, extract_dir)
    
    # Шаг 3: Подготовка подмножества
    total = prepare_subset(extract_dir, raw_dir, classes, args.max_files)
    
    # Шаг 4: Очистка (опционально)
    if not args.keep_archive:
        cleanup(extract_dir, keep_archive=True)
    
    print("\n" + "=" * 60)
    print("ГОТОВО!")
    print("=" * 60)
    print(f"Данные подготовлены в: {raw_dir}")
    print(f"Файлов: {total}")
    print(f"\nСледующие шаги:")
    print(f"  1. python code/preprocessing.py")
    print(f"  2. python code/split_data.py")
    print(f"  3. python code/train.py")
    print(f"\nИли запустите всё одной командой:")
    print(f"  python run_pipeline.py --full")


if __name__ == "__main__":
    main()