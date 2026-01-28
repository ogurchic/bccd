import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from pathlib import Path

from dataset import create_dataloaders
from train import AudioClassifier


def load_model(model_path, device):
    
    model = AudioClassifier(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def get_predictions(model, loader, device):

    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for spectrograms, labels in loader:
            spectrograms = spectrograms.to(device)
            
            # Получаем выход модели (logits)
            outputs = model(spectrograms)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return (
        np.array(all_labels),
        np.array(all_predictions),
        np.array(all_probabilities)
    )


def calculate_metrics(labels, predictions):

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def plot_confusion_matrix(labels, predictions, class_names, save_path=None):

    cm = confusion_matrix(labels, predictions)

    fig, ax = plt.subplots(figsize=(8, 6))
 
    im = ax.imshow(cm, cmap='Blues')
    plt.colorbar(im)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    ax.set_xlabel('Предсказано')
    ax.set_ylabel('Реально')
    ax.set_title('Confusion Matrix')

    for i in range(2):
        for j in range(2):
            # Выбираем цвет текста: белый на тёмном фоне, чёрный на светлом
            color = 'white' if cm[i, j] > cm.max() / 2 else 'black'

            ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                    color=color, fontsize=20)
    
    plt.tight_layout()
    
    # Сохраняем если указан путь
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix сохранена: {save_path}")
    
    plt.show()
    
    return cm


def print_classification_report(labels, predictions, class_names):

    report = classification_report(
        labels, 
        predictions, 
        target_names=class_names
    )
    
    print("\nПодробный отчёт:")
    print("-" * 50)
    print(report)


def evaluate_model(model_path, processed_dir, device):
    #Полная оценка модели на тестовых данных.
   
    print("=" * 50)
    print("ОЦЕНКА МОДЕЛИ")
    print("=" * 50)
    
    # Загружаем модель
    print(f"\nЗагрузка модели из {model_path}...")
    model = load_model(model_path, device)
    
    # Создаём DataLoader для теста
    print("Загрузка тестовых данных...")
    loaders = create_dataloaders(processed_dir, batch_size=16)
    test_loader = loaders['test']
    
    print(f"Тестовых примеров: {len(test_loader.dataset)}")
    
    # Получаем предсказания
    print("\nПолучение предсказаний...")
    labels, predictions, probabilities = get_predictions(model, test_loader, device)
    
    # Вычисляем метрики
    print("\nМетрики качества:")
    print("-" * 30)
    
    metrics = calculate_metrics(labels, predictions)
    
    for name, value in metrics.items():
        # :.4f — 4 знака после запятой
        # :.1% — проценты с 1 знаком (0.8333 → 83.3%)
        print(f"  {name.capitalize():12}: {value:.4f} ({value:.1%})")

    class_names = ['Cat', 'Dog']

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

        for idx in errors[:5]:
            true_class = class_names[labels[idx]]
            pred_class = class_names[predictions[idx]]
            confidence = probabilities[idx][predictions[idx]]
            
            print(f"    Пример {idx}: реально {true_class}, "
                  f"предсказано {pred_class} (уверенность: {confidence:.1%})")
    
    return metrics


if __name__ == "__main__":
    
    MODEL_PATH = "checkpoints/best_model.pt"
    PROCESSED_DIR = "data/processed"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}\n")
    
    metrics = evaluate_model(MODEL_PATH, PROCESSED_DIR, device)
    
    print("\n" + "=" * 50)
    print("Оценка завершена!")