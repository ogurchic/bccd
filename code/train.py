import csv
import datetime
import json
from pathlib import Path

import numpy as np
import torch
from config import cfg
from dataset import create_dataloaders
from torch import nn
from tqdm import tqdm

# Архитектура модели

class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm -> ReLU -> MaxPool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class AudioClassifier(nn.Module):
    def __init__(self, num_classes=None, use_attention=None):
        super().__init__()

        num_classes = num_classes if num_classes is not None else cfg.model.num_classes
        use_attention = use_attention if use_attention is not None else cfg.model.use_attention

        self.use_attention = use_attention

        # CNN encoder (без изменений)
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 256)

        # После 4 MaxPool высота уменьшается в 16 раз:
        # input: [1, n_mels=128, T]  →  output conv4: [256, 128/16=8, T/16]
        # Для LSTM flatten'им: 256 * 8 = 2048
        lstm_input_size = 256 * (cfg.audio.n_mels // 16)

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=cfg.model.lstm_hidden_size,
            num_layers=cfg.model.lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.model.lstm_dropout if cfg.model.lstm_num_layers > 1 else 0.0,
        )

        # Attention pooling (опционально)
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(cfg.model.lstm_hidden_size * 2, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )

        # FC head
        self.fc1 = nn.Linear(cfg.model.lstm_hidden_size * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(cfg.model.fc_dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B, 1, n_mels, T]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # [B, C, H, W] -> [B, W, C*H]
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        time_steps = x.size(1)
        x = x.reshape(batch_size, time_steps, -1)

        lstm_out, (hidden, _) = self.lstm(x)

        if self.use_attention:
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            x = (lstm_out * attn_weights).sum(dim=1)
        else:
            x = torch.cat([hidden[-2], hidden[-1]], dim=1)

        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Mixup

def mixup_data(x, y, alpha=0.2):
    """x' = lam * x + (1 - lam) * x_shuffled"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss для mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Обучение / Валидация

def train_epoch(model, loader, criterion, optimizer, device,
                use_mixup=True, mixup_alpha=0.2):
    """Одна эпоха обучения."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='  Train')
    for spectrograms, labels in pbar:
        spectrograms = spectrograms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_mixup:
            mixed_x, y_a, y_b, lam = mixup_data(spectrograms, labels, alpha=mixup_alpha)
            outputs = model(mixed_x)
            loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

            predictions = outputs.argmax(dim=1)
            correct += (lam * predictions.eq(y_a).sum().item()
                        + (1 - lam) * predictions.eq(y_b).sum().item())
        else:
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)

        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    """Валидация модели."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for spectrograms, labels in tqdm(loader, desc='  Val'):
            spectrograms = spectrograms.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    return total_loss / total, correct / total


def log_history(history, save_path):
    """Сохраняет историю обучения в CSV."""
    save_path = Path(save_path)
    with open(save_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_acc',
                         'val_loss', 'val_acc', 'lr'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_acc'][i]:.4f}",
                f"{history['val_loss'][i]:.6f}",
                f"{history['val_acc'][i]:.4f}",
                f"{history['lr'][i]:.6e}",
            ])
    print(f"  История обучения сохранена в {save_path}")


# Главный цикл

def train_model(model, loaders, criterion, optimizer, device,
                scheduler=None, num_epochs=None, save_dir=None):
    """Цикл обучения с Early Stopping и Scheduler."""
    num_epochs = num_epochs if num_epochs is not None else cfg.training.num_epochs
    save_dir = save_dir or cfg.paths.checkpoint_dir
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    patience_counter = 0
    patience = cfg.training.patience

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': [],
    }

    print(f"\n{'=' * 50}")
    print(f"НАЧАЛО ОБУЧЕНИЯ")
    print(f"{'=' * 50}")
    print(f"  Устройство:    {device}")
    print(f"  Эпох:          {num_epochs}")
    print(f"  Patience:      {patience}")
    print(f"  Mixup:         {cfg.training.use_mixup}")
    print(f"  Augmentation:  {cfg.training.use_augmentation}")
    print(f"  Attention:     {cfg.model.use_attention}")
    print(f"{'=' * 50}")

    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nЭпоха {epoch + 1}/{num_epochs}  |  LR: {current_lr:.2e}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, loaders['train'], criterion, optimizer, device,
            use_mixup=cfg.training.use_mixup,
            mixup_alpha=cfg.training.mixup_alpha,
        )
        val_loss, val_acc = validate(model, loaders['val'], criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        print(f"  Train → Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / 'best_model.pt')
            print(f"  ✓ Новая лучшая модель сохранена (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"  ✗ Нет улучшения ({patience_counter}/{patience})")

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if patience_counter >= patience:
            print(f"\n⚠ Early stopping на эпохе {epoch + 1}")
            break

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), save_path / 'last_model.pt')

    print(f"\n{'=' * 50}")
    print(f"ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print(f"{'=' * 50}")
    print(f"  Лучший val_loss: {best_val_loss:.4f}")

    log_history(history, save_path / 'history.csv')

    # Метаданные модели — нужны для predict.py и evaluate.py
    model_meta = {
        'num_classes': cfg.model.num_classes,
        'use_attention': cfg.model.use_attention,
        'max_frames': cfg.audio.max_frames,
        'sample_rate': cfg.audio.sample_rate,
        'n_mels': cfg.audio.n_mels,
        'best_val_loss': best_val_loss,
        'trained_at': str(datetime.datetime.now()),
    }
    with open(save_path / 'model_meta.json', 'w', encoding='utf-8') as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)
    print(f"  Метаданные модели: {save_path / 'model_meta.json'}")

    return history


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Загрузка DataLoader'ов...")
    loaders = create_dataloaders()

    print("Инициализация модели...")
    model = AudioClassifier().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Всего параметров:      {total_params:,}")
    print(f"  Обучаемых параметров:  {trainable_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    scheduler = None
    if cfg.training.use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min',
            factor=cfg.training.scheduler_factor,
            patience=cfg.training.scheduler_patience,
            min_lr=cfg.training.min_lr,
            verbose=True,
        )

    history = train_model(
        model=model, loaders=loaders, criterion=criterion,
        optimizer=optimizer, device=device, scheduler=scheduler,
    )