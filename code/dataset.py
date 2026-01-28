import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path



class AudioDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        
        self.data = pd.read_csv(csv_path)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        spectrogram_path = row['path']
        label = row['label']
        
        spectrogram = np.load(spectrogram_path)
        spectrogram = np.expand_dims(spectrogram, axis=0)

        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        
        return spectrogram, label

def create_dataloaders(processed_dir, batch_size=16):
    processed_path = Path(processed_dir)

    train_dataset = AudioDataset(processed_path / 'train.csv')
    val_dataset = AudioDataset(processed_path / 'val.csv')
    test_dataset = AudioDataset(processed_path / 'test.csv')
    
    use_cuda = torch.cuda.is_available()
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda
    )

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def check_dataloaders(loaders):
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
        print(f"    Мин/макс спектрограмм: {spectrograms.min().item():.3f} / {spectrograms.max().item():.3f}")
        print(f"    Уникальные метки: {torch.unique(labels).tolist()}")
        print()


if __name__ == "__main__":
    
    PROCESSED_DIR = "data/processed"
    BATCH_SIZE = 16
    
    print("Создание DataLoader'ов...\n")
    
    loaders = create_dataloaders(
        processed_dir=PROCESSED_DIR,
        batch_size=BATCH_SIZE
    )

    check_dataloaders(loaders)
    
    print("Пример итерации по train_loader:")
    print("-" * 40)

    for batch_idx, (spectrograms, labels) in enumerate(loaders['train']):
        print(f"  Batch {batch_idx}: spectrograms {spectrograms.shape}, labels {labels.shape}")

        if batch_idx >= 2:
            print("  ...")
            break
    
    print(f"\n  Всего batch'ей: {len(loaders['train'])}")