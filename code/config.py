"""
Единый конфиг проекта BCCD.
Поддерживает датасеты:
- Собственный (cat/dog)
- Speech Commands v2 (до 35 команд)
"""
from dataclasses import dataclass, field
from typing import List
from pathlib import Path


@dataclass
class AudioConfig:
    """Параметры обработки аудио и мел-спектрограмм."""
    sample_rate: int = 16000          # ← Speech Commands использует 16kHz!
    n_mels: int = 128
    n_fft: int = 1024                 # ← меньше для 16kHz
    hop_length: int = 256             # ← меньше для 16kHz
    max_duration_sec: float = 1.0     # ← Speech Commands ~1 сек

    @property
    def max_frames(self) -> int:
        return int(self.max_duration_sec * self.sample_rate / self.hop_length) + 1


@dataclass
class ModelConfig:
    """Параметры архитектуры модели."""
    num_classes: int = 10             # ← 10 команд по умолчанию
    use_attention: bool = False
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    fc_dropout: float = 0.3

    class_names: List[str] = field(default_factory=lambda: [
        'yes', 'no', 'up', 'down', 'left', 'right', 'stop', 'go', 'cat', 'dog'
    ])


@dataclass
class TrainingConfig:
    """Параметры обучения."""
    batch_size: int = 32              # ← можно больше для 16kHz
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 50
    patience: int = 10

    use_scheduler: bool = True
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    use_augmentation: bool = True
    use_mixup: bool = True
    mixup_alpha: float = 0.2


@dataclass
class PathConfig:
    """Пути к данным и артефактам."""
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"


@dataclass
class Config:
    """Корневой конфиг."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @property
    def class_map_path(self) -> Path:
        return Path(self.paths.processed_dir) / 'class_map.json'

    @property
    def metadata_path(self) -> Path:
        return Path(self.paths.processed_dir) / 'metadata.csv'


cfg = Config()