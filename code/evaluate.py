"""
Оценка качества модели на тестовой выборке:
метрики, confusion matrix, classification report.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from config import cfg
from dataset import create_dataloaders
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from train import AudioClassifier


def load_model(model_path, device, num_classes=None):
    """Загружает обученную модель из checkpoint."""
    num_classes = num_classes if num_classes is not None else cfg.model.num_classes
    model = AudioClassifier(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_predictions(model, loader, device):
    """Прогоняет DataLoader через модель."""
    all_labels, all_predictions, all_probabilities = [], [], []

    with torch.no_grad():
        for spectrograms, labels in loader:
            spectrograms = spectrograms.to(device)
            outputs = model(spectrograms)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def calculate_metrics(labels, predictions, average='weighted'):
    """Считает основные метрики классификации."""
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average=average, zero_division=0),
        'recall': recall_score(labels, predictions, average=average, zero_division=0),
        'f1': f1_score(labels, predictions, average=average, zero_division=0),
    }


def load_class_names(processed_dir=None):
    """Загружает список имён классов из class_map.json."""
    processed_dir = processed_dir or cfg.paths.processed_dir
    class_map_path = Path(processed_dir) / 'class_map.json'
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    return [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):
    """Рисует confusion matrix для произвольного числа классов."""
    cm = confusion_matrix(labels, predictions)
    n_classes = len(class_names)

    fig, ax = plt.subplots(
        figsize=(max(8, n_classes * 0.6), max(6, n_classes * 0.5)),
    )
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Реально')
    ax.set_title('Confusion Matrix')

    for i in range(n_classes):
        for j in range(n_classes):
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', color=color,
                    fontsize=max(8, 14 - n_classes))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  Confusion matrix сохранена: {save_path}")

    plt.show()
    return cm


def print_classification_report(labels, predictions, class_names):
    """Выводит подробный classification report."""
    report = classification_report(
        labels, predictions,
        target_names=class_names, zero_division=0,
    )
    print("\nПодробный отчёт:")
    print("-" * 50)
    print(report)


def evaluate_model(model_path, processed_dir=None, device=None):
    """Полная оценка модели на тестовых данных."""
    processed_dir = processed_dir or cfg.paths.processed_dir
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 50)

    class_names = load_class_names(processed_dir)
    print(f"Классов: {len(class_names)} → {class_names}")

    print(f"\nЗагрузка модели из {model_path}...")
    model = load_model(model_path, device)

    print("Загрузка тестовых данных...")
    loaders = create_dataloaders(processed_dir=processed_dir)
    test_loader = loaders['test']
    print(f"Тестовых примеров: {len(test_loader.dataset)}")

    print("\nПолучение предсказаний...")
    labels, predictions, probabilities = get_predictions(model, test_loader, device)

    print("\nМетрики качества:")
    print("-" * 30)
    metrics = calculate_metrics(labels, predictions)
    for name, value in metrics.items():
        print(f"  {name.capitalize():12}: {value:.4f} ({value:.1%})")

    print_classification_report(labels, predictions, class_names)

    print("\nConfusion Matrix:")
    print("-" * 30)
    save_path = Path(processed_dir) / 'confusion_matrix.png'
    cm = plot_confusion_matrix(labels, predictions, class_names, save_path)

    print("\nАнализ ошибок:")
    print("-" * 30)
    errors = np.where(labels != predictions)[0]
    print(f"  Всего ошибок: {len(errors)} из {len(labels)}")

    if len(errors) > 0:
        print("\n  Примеры ошибок:")
        data_df = test_loader.dataset.data   # <- DataFrame датасета (порядок = порядку меток)
        for idx in errors[:10]:
            true_class = class_names[labels[idx]]
            pred_class = class_names[predictions[idx]]
            confidence = probabilities[idx][predictions[idx]]
             # Имя исходного файла
            row = data_df.iloc[idx]
            if 'source_path' in data_df.columns:
                fname = Path(row['source_path']).name
            else:
                fname = row['filename']

            print(f"    {fname}: реально {true_class}, "
                  f"предсказано {pred_class} (уверенность: {confidence:.1%})")

    return metrics


if __name__ == "__main__":
    MODEL_PATH = "checkpoints/best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}\n")

    metrics = evaluate_model(MODEL_PATH)

    print("\n" + "=" * 50)
    print("Оценка завершена!")