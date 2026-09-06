import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "code"))

import json

import librosa
import numpy as np
import torch
from config import cfg
from train import AudioClassifier
from preprocessing import find_wav_files


def load_model(model_path, device, num_classes=None):
    """Загружает обученную модель."""
    num_classes = num_classes if num_classes is not None else cfg.model.num_classes
    model = AudioClassifier(num_classes=num_classes)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def preprocess_audio(audio_path):
    """
    Преобразует аудиофайл в мел-спектрограмму с теми же параметрами,
    что использовались при обучении (cfg.audio).
    """
    waveform, _ = librosa.load(
        audio_path, sr=cfg.audio.sample_rate, mono=True,
    )

    mel_spec = librosa.feature.melspectrogram(
        y=waveform, sr=cfg.audio.sample_rate,
        n_mels=cfg.audio.n_mels, n_fft=cfg.audio.n_fft,
        hop_length=cfg.audio.hop_length,
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    spec_min = mel_spec_db.min()
    spec_max = mel_spec_db.max()
    if spec_max - spec_min > 1e-8:
        mel_spec_norm = (mel_spec_db - spec_min) / (spec_max - spec_min)
    else:
        mel_spec_norm = np.zeros_like(mel_spec_db)

    max_frames = cfg.audio.max_frames
    if mel_spec_norm.shape[1] > max_frames:
        mel_spec_norm = mel_spec_norm[:, :max_frames]
    elif mel_spec_norm.shape[1] < max_frames:
        pad_width = max_frames - mel_spec_norm.shape[1]
        mel_spec_norm = np.pad(mel_spec_norm, ((0, 0), (0, pad_width)),
                               mode='constant')

    return mel_spec_norm


def load_class_names(processed_dir=None):
    """Загружает имена классов из class_map.json."""
    processed_dir = processed_dir or cfg.paths.processed_dir
    class_map_path = Path(processed_dir) / 'class_map.json'
    if not class_map_path.exists():
        raise FileNotFoundError(
            f"Не найден class_map.json: {class_map_path}. "
            f"Сначала запустите preprocessing.py"
        )
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    return [k for k, v in sorted(class_map.items(), key=lambda x: x[1])]


def predict(model, audio_path, class_names, device):
    """Делает предсказание для одного аудиофайла."""
    spectrogram = preprocess_audio(audio_path)
    spectrogram = np.expand_dims(spectrogram, axis=0)  # channel dim
    spectrogram = np.expand_dims(spectrogram, axis=0)  # batch dim
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(spectrogram)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()

    probs = probabilities[0].cpu().numpy()

    return {
        'class': class_names[predicted_class],
        'confidence': float(probs[predicted_class] * 100),
        'probabilities': {
            class_names[i]: float(p * 100) for i, p in enumerate(probs)
        },
    }


def predict_file(audio_path, model_path=None):
    """Предсказание для одного файла."""
    model_path = model_path or str(Path(cfg.paths.checkpoint_dir) / 'best_model.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    class_names = load_class_names()
    return predict(model, audio_path, class_names, device)


def predict_folder(folder_path, model_path=None):
    """Предсказание для всех .wav файлов в папке."""
    model_path = model_path or str(Path(cfg.paths.checkpoint_dir) / 'best_model.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    class_names = load_class_names()

    folder = Path(folder_path)
    audio_files = find_wav_files(folder)

    results = {}
    for audio_file in audio_files:
        try:
            result = predict(model, audio_file, class_names, device)
            results[audio_file.name] = result
        except Exception as e:
            results[audio_file.name] = {'error': str(e)}
    return results


# CLI

if __name__ == "__main__":
    MODEL_PATH = str(Path(cfg.paths.checkpoint_dir) / 'best_model.pt')

    if len(sys.argv) < 2:
        print("Использование:")
        print("  python predict.py <путь_к_аудио.wav>")
        print("  python predict.py <путь_к_папке>")
        print()
        print("Примеры:")
        print("  python predict.py my_cat.wav")
        print("  python predict.py sounds/")
        sys.exit(1)

    input_path = Path(sys.argv[1])

    if not input_path.exists():
        print(f"❌ Ошибка: путь не найден: {input_path}")
        sys.exit(1)

    print("=" * 50)
    print("ПРЕДСКАЗАНИЕ")
    print("=" * 50)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    print(f"Загрузка модели из {MODEL_PATH}...")
    model = load_model(MODEL_PATH, device)
    class_names = load_class_names()
    print(f"Классов: {len(class_names)} → {class_names}")
    print()

    if input_path.is_file():
        print(f"Файл: {input_path}")
        print("-" * 30)
        result = predict(model, input_path, class_names, device)
        print(f"Предсказание: {result['class']}")
        print(f"Уверенность: {result['confidence']:.1f}%")
        print()
        print("Вероятности:")
        for cls, prob in result['probabilities'].items():
            print(f"  {cls:20}: {prob:.1f}%")

    elif input_path.is_dir():
        audio_files = find_wav_files(input_path)
        if not audio_files:
            print(f"❌ В папке {input_path} нет .wav файлов")
            sys.exit(1)

        print(f"Папка: {input_path}")
        print(f"Найдено файлов: {len(audio_files)}")
        print("-" * 30)
        print()

        max_name_len = max(len(f.name) for f in audio_files)
        max_class_len = max(len(c) for c in class_names)

        for audio_file in audio_files:
            try:
                result = predict(model, audio_file, class_names, device)
                filename = audio_file.name
                pred_class = result['class']
                confidence = result['confidence']
                print(f"  {filename:<{max_name_len}}  →  "
                      f"{pred_class:<{max_class_len}}  ({confidence:.1f}%)")
            except Exception as e:
                print(f"  {audio_file.name:<{max_name_len}}  →  ❌ Ошибка: {e}")

    print()
    print("=" * 50)
    print("Готово!")