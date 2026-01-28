import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

import torch
import numpy as np
import librosa

from train import AudioClassifier


def load_model(model_path, device):

    model = AudioClassifier(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def preprocess_audio(audio_path, sr=22050, n_mels=128, n_fft=2048, hop_length=512):
    # Загружаем аудио
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # Создаём мел-спектрограмму
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Переводим в децибелы
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Нормализуем в [0, 1]
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    return mel_spec_norm


def predict(model, audio_path, device):
   
    # Названия классов (в том же порядке, как при обучении)
    class_names = ['Cat', 'Dog']
    
    # Предобработка аудио
    spectrogram = preprocess_audio(audio_path)
    
    # Добавляем размерность канала: [128, T] → [1, 128, T]
    spectrogram = np.expand_dims(spectrogram, axis=0)
    
    # Добавляем размерность batch: [1, 128, T] → [1, 1, 128, T]
    spectrogram = np.expand_dims(spectrogram, axis=0)
    
    # Преобразуем в тензор
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
    
    # Перемещаем на устройство
    spectrogram = spectrogram.to(device)
    

    with torch.no_grad():
        outputs = model(spectrogram)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
    
    # Извлекаем вероятности
    probs = probabilities[0].cpu().numpy()
    
    result = {
        'class': class_names[predicted_class],
        'confidence': probs[predicted_class] * 100,
        'probabilities': {
            'Cat': probs[0] * 100,
            'Dog': probs[1] * 100
        }
    }
    
    return result


def predict_file(audio_path, model_path="checkpoints/best_model.pt"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    result = predict(model, audio_path, device)
    
    return result


def predict_folder(folder_path, model_path="checkpoints/best_model.pt"):

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    
    folder = Path(folder_path)
    audio_files = list(folder.glob("*.wav"))
    
    results = {}
    
    for audio_file in audio_files:
        try:
            result = predict(model, audio_file, device)
            results[audio_file.name] = result
        except Exception as e:
            results[audio_file.name] = {'error': str(e)}
    
    return results


if __name__ == "__main__":
    
    import sys
    
    MODEL_PATH = "checkpoints/best_model.pt"
    
    # Проверяем аргументы командной строки
    if len(sys.argv) < 2:
        print("Использование:")
        print("  python src/predict.py <путь_к_аудио.wav>")
        print("  python src/predict.py <путь_к_папке>")
        print()
        print("Примеры:")
        print("  python src/predict.py my_cat.wav")
        print("  python src/predict.py sounds/")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    
    if not input_path.exists():
        print(f"Ошибка: путь не найден: {input_path}")
        sys.exit(1)
    
    print("=" * 50)
    print("ПРЕДСКАЗАНИЕ")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    
    # Загружаем модель один раз
    print(f"Загрузка модели из {MODEL_PATH}...")
    model = load_model(MODEL_PATH, device)
    print()
    
    if input_path.is_file():
        print(f"Файл: {input_path}")
        print("-" * 30)
        
        result = predict(model, input_path, device)
        
        print(f"Предсказание: {result['class']}")
        print(f"Уверенность:  {result['confidence']:.1f}%")
        print()
        print("Вероятности:")
        print(f"  Cat: {result['probabilities']['Cat']:.1f}%")
        print(f"  Dog: {result['probabilities']['Dog']:.1f}%")
    
    elif input_path.is_dir():
        audio_files = list(input_path.glob("*.wav"))
        
        if not audio_files:
            print(f"В папке {input_path} нет .wav файлов")
            sys.exit(1)
        
        print(f"Папка: {input_path}")
        print(f"Найдено файлов: {len(audio_files)}")
        print("-" * 30)
        print()
        
        for audio_file in audio_files:
            try:
                result = predict(model, audio_file, device)
                
                # Красивый вывод с выравниванием
                filename = audio_file.name
                pred_class = result['class']
                confidence = result['confidence']
                
                print(f"{filename:30} → {pred_class:4} ({confidence:.1f}%)")
                
            except Exception as e:
                print(f"{audio_file.name:30} → Ошибка: {e}")
        
        print()
    
    print("=" * 50)
    print("Готово!")