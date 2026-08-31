"""
Предобработка сырых аудиофайлов: конвертация в мел-спектрограммы,
фиксация длины через padding/truncation и автоматическое определение классов.
"""
import json
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import cfg
from tqdm import tqdm


def audio_to_melspec(
    path,
    sr=None,
    n_mels=None,
    n_fft=None,
    hop_length=None,
    max_frames=None,
):
    """
    Преобразует аудиофайл в нормализованную мел-спектрограмму фиксированной длины.
    """
    sr = sr or cfg.audio.sample_rate
    n_mels = n_mels or cfg.audio.n_mels
    n_fft = n_fft or cfg.audio.n_fft
    hop_length = hop_length or cfg.audio.hop_length
    max_frames = max_frames or cfg.audio.max_frames

    waveform, _ = librosa.load(path, sr=sr, mono=True)

    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    spec_min = mel_spec_db.min()
    spec_max = mel_spec_db.max()
    if spec_max - spec_min > 1e-8:
        mel_spec_norm = (mel_spec_db - spec_min) / (spec_max - spec_min)
    else:
        mel_spec_norm = np.zeros_like(mel_spec_db)

    # Фиксация длины по временной оси
    if mel_spec_norm.shape[1] > max_frames:
        mel_spec_norm = mel_spec_norm[:, :max_frames]
    elif mel_spec_norm.shape[1] < max_frames:
        pad_width = max_frames - mel_spec_norm.shape[1]
        mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)), mode='constant')

    return mel_spec_norm

def find_wav_files(folder):
    """
    Ищет .wav файлы в папке (регистр расширения не важен).
    В отличие от glob('*.wav') + glob('*.WAV'), не даёт
    дубликатов на Windows.
    """
    folder = Path(folder)
    return sorted(
        p for p in folder.glob('*')
        if p.is_file() and p.suffix.lower() == '.wav'
    )

def process_dataset(raw_dir=None, processed_dir=None, sr=None):
    """
    Обрабатывает все аудиофайлы в raw_dir.
    Автоматически находит все подпапки (классы).
    
    Сохраняет в метаданных:
    - path: путь к .npy спектрограмме (для обучения)
    - source_path: путь к исходному .wav (для предсказания)
    """
    raw_dir = raw_dir or cfg.paths.raw_dir
    processed_dir = processed_dir or cfg.paths.processed_dir
    sr = sr or cfg.audio.sample_rate

    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    # Автоматическое определение классов
    class_names = sorted([
        d.name for d in raw_path.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    if not class_names:
        raise ValueError(f"В папке {raw_path} не найдено ни одной подпапки с классами")

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    print(f"Найдено классов ({len(class_names)}): {class_names}")
    print(f"Маппинг: {class_to_idx}")

    metadata = []
    for class_name in class_names:
        class_dir = raw_path / class_name
        (processed_path / class_name).mkdir(parents=True, exist_ok=True)
        audio_files = find_wav_files(class_dir)
        print(f"\nОбработка класса '{class_name}': {len(audio_files)} файлов")

        for audio_file in tqdm(audio_files, desc=class_name):
            try:
                mel_spec = audio_to_melspec(audio_file, sr=sr)
                save_name = audio_file.stem + '.npy'
                save_path = processed_path / class_name / save_name
                np.save(save_path, mel_spec)
                metadata.append({
                    'filename': save_name,
                    'class': class_name,
                    'label': class_to_idx[class_name],
                    'path': str(save_path),           # путь к .npy (для обучения)
                    'source_path': str(audio_file),   # путь к .wav (для предсказания) ← НОВОЕ
                })
            except Exception as e:
                print(f"  ⚠ Ошибка обработки {audio_file.name}: {e}")

    df = pd.DataFrame(metadata)
    df.to_csv(processed_path / 'metadata.csv', index=False)

    with open(processed_path / 'class_map.json', 'w', encoding='utf-8') as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Готово! Обработано {len(metadata)} файлов")
    print(f"   Метаданные: {processed_path / 'metadata.csv'}")
    print(f"   Class map:  {processed_path / 'class_map.json'}")

    return df, class_names


def visualize_spec(processed_dir=None, num_samples=5):
    """Визуализирует примеры мел-спектрограмм для каждого класса."""
    processed_dir = processed_dir or cfg.paths.processed_dir
    processed_path = Path(processed_dir)

    with open(processed_path / 'class_map.json', 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    class_names = sorted(class_map.keys())

    n_classes = len(class_names)
    fig, axes = plt.subplots(
        n_classes, num_samples,
        figsize=(15, 3 * n_classes), squeeze=False,
    )

    for row, class_name in enumerate(class_names):
        class_dir = processed_path / class_name
        files = sorted(class_dir.glob('*.npy'))[:num_samples]
        for col, file in enumerate(files):
            mel_spec = np.load(file)
            axes[row, col].imshow(
                mel_spec, aspect='auto', origin='lower', cmap='viridis',
            )
            axes[row, col].set_title(f'{class_name}\n{file.stem}', fontsize=8)
            axes[row, col].set_xlabel('Время')
            axes[row, col].set_ylabel('Частота')

    plt.tight_layout()
    save_path = processed_path / 'spectrograms_preview.png'
    plt.savefig(save_path, dpi=150)
    # НЕ вызываем plt.show() при использовании Agg backend
    try:
        plt.show()
    except Exception:
        pass  # Игнорируем при неинтерактивном backend
    print(f"Превью сохранено в {save_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("ПРЕДОБРАБОТКА ДАТАСЕТА")
    print("=" * 50)
    print(f"Source:       {cfg.paths.raw_dir}")
    print(f"Target:       {cfg.paths.processed_dir}")
    print(f"Sample rate:  {cfg.audio.sample_rate} Hz")
    print(f"Max duration: {cfg.audio.max_duration_sec} сек")
    print(f"Max frames:   {cfg.audio.max_frames}")
    print("=" * 50)

    metadata, class_names = process_dataset()

    print("\nСоздаю визуализацию...")
    visualize_spec(num_samples=5)