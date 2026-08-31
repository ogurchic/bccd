from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
from config import cfg
from torch.utils.data import DataLoader, Dataset


class AudioDataset(Dataset):
    """
    Dataset спектрограмм с опциональной аугментацией:
    - Time masking
    - Frequency masking
    - Gaussian noise
    - Time shift
    """
    def __init__(self, csv_path, augment=False):
        self.data = pd.read_csv(csv_path)
        self.augment = augment
        self.max_frames = cfg.audio.max_frames

    def __len__(self):
        return len(self.data)

    def _apply_augmentation(self, spectrogram):
        """Применяет случайные аугментации к спектрограмме [C, F, T]."""
        # Time masking
        if np.random.random() < 0.5:
            T = spectrogram.shape[2]
            max_t_mask = max(1, T // 10)
            t_mask_len = np.random.randint(1, max_t_mask + 1)
            t0 = np.random.randint(0, max(1, T - t_mask_len))
            spectrogram[:, :, t0:t0 + t_mask_len] = 0.0

        # Frequency masking
        if np.random.random() < 0.5:
            F = spectrogram.shape[1]
            max_f_mask = max(1, F // 10)
            f_mask_len = np.random.randint(1, max_f_mask + 1)
            f0 = np.random.randint(0, max(1, F - f_mask_len))
            spectrogram[:, f0:f0 + f_mask_len, :] = 0.0

        # Gaussian noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, spectrogram.shape).astype(np.float32)
            spectrogram = np.clip(spectrogram + noise, 0.0, 1.0)

        # Time shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-self.max_frames // 10, self.max_frames // 10 + 1)
            spectrogram = np.roll(spectrogram, shift, axis=2)

        return spectrogram

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        spectrogram_path = row['path']
        label = int(row['label'])

        spectrogram = np.load(spectrogram_path).astype(np.float32)

        # Добавляем размерность канала: [F, T] → [1, F, T]
        if spectrogram.ndim == 2:
            spectrogram = np.expand_dims(spectrogram, axis=0)

        if self.augment:
            spectrogram = self._apply_augmentation(spectrogram)

        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return spectrogram, label


import os  # ← добавь в импорты наверху файла

def create_dataloaders(processed_dir=None, batch_size=None, num_workers=None):
    """
    Создаёт DataLoader'ы для train/val/test.
    
    ИЗМЕНЕНИЯ для ускорения на GPU:
    - num_workers > 0: данные читаются параллельно с диска, GPU не ждёт
    - persistent_workers: воркеры не пересоздаются каждую эпоху
    - prefetch_factor: воркеры заранее готовят следующие батчи
    - pin_memory: быстрая передача на GPU
    """
    processed_dir = processed_dir or cfg.paths.processed_dir
    batch_size = batch_size or cfg.training.batch_size
    processed_path = Path(processed_dir)

    use_cuda = torch.cuda.is_available()

    # Автоподбор числа воркеров: половина ядер CPU, но не больше 8
    if num_workers is None:
        num_workers = getattr(cfg.training, 'num_workers', 0)
        if num_workers == 0 and use_cuda:
            num_workers = min(os.cpu_count() // 2, 8)

    train_dataset = AudioDataset(
        processed_path / 'train.csv',
        augment=cfg.training.use_augmentation,
    )
    val_dataset = AudioDataset(processed_path / 'val.csv', augment=False)
    test_dataset = AudioDataset(processed_path / 'test.csv', augment=False)

    # Общие параметры для всех loader'ов
    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    if num_workers > 0:
        common_kwargs['persistent_workers'] = True   # не убивать воркеров между эпохами
        common_kwargs['prefetch_factor'] = 4         # предзагрузка 4 батчей на воркер

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **common_kwargs,
    )

    print(f"  DataLoader: batch_size={batch_size}, num_workers={num_workers}")

    return {'train': train_loader, 'val': val_loader, 'test': test_loader}

def check_dataloaders(loaders):
    """Проверяет корректность работы DataLoader'ов."""
    print("Проверка DataLoader'ов:\n")
    for name, loader in loaders.items():
        print(f"  {name}:")
        print(f"    Количество batch'ей: {len(loader)}")
        print(f"    Размер датасета: {len(loader.dataset)}")

        spectrograms, labels = next(iter(loader))
        print(f"    Форма спектрограмм: {spectrograms.shape}")
        print(f"    Форма меток: {labels.shape}")
        print(f"    Тип спектрограмм: {spectrograms.dtype}")
        print(f"    Тип меток: {labels.dtype}")
        print(f"    Мин/макс спектрограмм: "
              f"{spectrograms.min().item():.3f} / {spectrograms.max().item():.3f}")
        print(f"    Уникальные метки: {torch.unique(labels).tolist()}")
        print()


if __name__ == "__main__":
    print("Создание DataLoader'ов...\n")
    loaders = create_dataloaders()
    check_dataloaders(loaders)

    print("Пример итерации по train_loader:")
    print("-" * 40)
    for batch_idx, (spectrograms, labels) in enumerate(loaders['train']):
        print(f"  Batch {batch_idx}: spectrograms {spectrograms.shape}, "
              f"labels {labels.shape}")
        if batch_idx >= 2:
            print("  ...")
            break
    print(f"\n  Всего batch'ей: {len(loaders['train'])}")