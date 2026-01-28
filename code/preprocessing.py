import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def audio_to_melspec(path, sr=44100, n_mels=128, n_fft=2048, hop_leght=512):
    waveform, _ = librosa.load(path, sr=sr, mono=True)
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_leght
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    #(x - min) / (max - min)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    return mel_spec_norm

def process_dataset(raw_dir, processed_dir, sr=44100):
    raw_path = Path(raw_dir)
    processed_path = Path(processed_dir)

    for class_name in ['cat', 'dog']:
        (processed_path / class_name).mkdir(parents=True, exist_ok=True)
    
    metadata = []

    for class_name in ['cat', 'dog']:
        class_dir = raw_path / class_name
        audio_files = list(class_dir.glob('*.wav'))
        print(f"\nОбработка класса '{class_name}': {len(audio_files)} файлов")

        for audio_file in tqdm(audio_files):
            mel_spec = audio_to_melspec(audio_file, sr=sr)

            save_name = audio_file.stem + '.npy'
            save_path = processed_path / class_name / save_name

            np.save(save_path, mel_spec)

            metadata.append({
                'filename': save_name,
                'class': class_name,
                'label': 0 if class_name == 'cat' else 1,
                'path': str(save_path)
            })

    df = pd.DataFrame(metadata)
    df.to_csv(processed_path / 'metadata.csv', index=False)
    
    print(f"\nГотово! Обработано {len(metadata)} файлов")
    print(f"Метаданные сохранены в {processed_path / 'metadata.csv'}")
    return df

def visualize_spec(processed_dir, num_samples=5):
    processed_path = Path(processed_dir)
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    for row, class_name in enumerate(['cat', 'dog']):
        
        class_dir = processed_path / class_name

        files = list(class_dir.glob('*.npy'))[:num_samples]
        
        for col, file in enumerate(files):

            mel_spec = np.load(file)
            
            axes[row, col].imshow(
                mel_spec,
                aspect='auto',
                origin='lower',
                cmap='viridis'
            )
            
            axes[row, col].set_title(f'{class_name}\n{file.stem}', fontsize=8)
            axes[row, col].set_xlabel('Время')
            axes[row, col].set_ylabel('Частота')

    plt.tight_layout()
    plt.savefig(processed_path / 'spectrograms_preview.png', dpi=150)
    plt.show()
    print(f"Превью сохранено в {processed_path / 'spectrograms_preview.png'}")


if __name__ == "__main__":

    RAW_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    SAMPLE_RATE = 44100

    print("Начинаю обработку датасета...")
    metadata = process_dataset(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        sr=SAMPLE_RATE
    )
    print("\nСоздаю визуализацию...")
    visualize_spec(PROCESSED_DIR, num_samples=5)