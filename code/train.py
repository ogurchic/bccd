import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm

from dataset import create_dataloaders


class ConvBlock(nn.Module):
    
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
    
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.conv3 = ConvBlock(64, 128) # надо поиграться с количеством слоёв
        self.conv4 = ConvBlock(128, 256)
        
        self.lstm = nn.LSTM(
            input_size=128 * 16,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 # нало поиграться с количеством 
        )
        
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)  # (batch, channels, height, width) -> (batch, width, channels, height)
        time_steps = x.size(1)
        x = x.reshape(batch_size, time_steps, -1)  # (batch, time, features)
        
        _, (hidden, _) = self.lstm(x)
        x = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0    # сумма всех loss'ов
    correct = 0         # количество правильных предсказаний
    total = 0           # общее количество примеров
    
    for spectrograms, labels in tqdm(loader, desc='Training'):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        total += labels.size(0)
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
    
    return total_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for spectrograms, labels in tqdm(loader, desc='Validation'):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
    
    return total_loss / total, correct / total


def train_model(model, loaders, criterion, optimizer, device, num_epochs=20, save_dir='checkpoints'):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    best_val_loss = float('inf')
    patience = 30 # поиграть
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"\nНачинаю обучение на {device}")
    print(f"Эпох: {num_epochs}, Patience: {patience}")
    print("=" * 50)
    
    for epoch in range(num_epochs):
        print(f"\nЭпоха {epoch + 1}/{num_epochs}")
        print("-" * 30)
        
        train_loss, train_acc = train_epoch(model, loaders['train'], criterion, optimizer, device)
        val_loss, val_acc = validate(model, loaders['val'], criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path / 'best_model.pt')
            print(f"  ✓ Модель сохранена")
        else:
            patience_counter += 1
            print(f"  Нет улучшения ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping!")
                break
    
    print("\n" + "=" * 50)
    print("Обучение завершено!")
    print(f"Лучший val_loss: {best_val_loss:.4f}")
    
    return history


if __name__ == "__main__":
    
    PROCESSED_DIR = "data/processed"
    BATCH_SIZE = 16
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    SAVE_DIR = "checkpoints"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    
    loaders = create_dataloaders(PROCESSED_DIR, BATCH_SIZE)
    model = AudioClassifier(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    history = train_model(
        model=model,
        loaders=loaders,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=NUM_EPOCHS,
        save_dir=SAVE_DIR
    )