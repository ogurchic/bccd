# BCCD — распознование голосовых команд (изначально был Binary Classifier Cat-Dog) 

Нейронная сеть для классификации звуков животных (кошек и собак) на основе анализа аудиосигналов.


Проект реализует гибридную архитектуру CNN + LSTM для классификации аудиофайлов. Звуковые файлы преобразуются в мел-спектрограммы, которые затем анализируются нейронной сетью.

Датасет: [Speech Commands v2](http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz) (Google).

## Распознаваемые команды

`yes` `no` `up` `down` `left` `right` `stop` `go` `on` `off`

Классы определяются автоматически по подпапкам в `data/raw/` — можно добавить
любые свои слова из датасета

## Возможности

- 🧠 CNN (4×ConvBlock) + двунаправленный LSTM, опциональный attention-pooling
- 🔁 SpecAugment, Mixup, Early Stopping, ReduceLROnPlateau
- ⚡ Оптимизация под GPU: параллельная загрузка данных, большие батчи, pin_memory
- 📊 Полная оценка: accuracy/precision/recall/F1, classification report, confusion matrix, разбор ошибок с именами файлов
- 🎙 Реалтайм-режим: непрерывная запись с микрофона, триггер по слову, статистика и лог сессии
- ⚙️ Единый конфиг проекта — `code/config.py`
- 🖥 Работает и на CPU, и на CUDA

## Установка

1. Клонируйте репозиторий или скачайте проект

2. Создайте виртуальное окружение:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```
    Для работы CUDA (укажите подходящую вам версию CUDA, у меня 12.1)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```    

## Быстрый старт

# 1. Скачать и сбалансировать датасет (10 классов)
python code/install_speech_commands.py --balance

# 2. Прогнать весь пайплайн: предобработка → split → обучение → оценка → демо-предсказание
python run_pipeline.py --full
Без --full запускается быстрый тестовый прогон (мало эпох, без аугментаций), если нет ГП лучше запускать так

## Использование по шагам
python code/preprocessing.py     # WAV → спектрограммы
python code/split_data.py        # разбиение 70/15/15
python code/train.py             # обучение
python code/evaluate.py          # оценка на тесте

# Предсказание на файлах
python predict.py my_voice.wav        # один файл
python predict.py sounds/             # папка с файлами

## Реалтайм-распознавание с микрофона
# 1. Калибровка: посмотреть уровень громкости и подобрать ENERGY_THRESHOLD
python code/mic_check.py

# 2. Запуск детектора
python live_detect.py

вывод должен выглядеть примерно так:
```  
🔔 [00:15] stop   | уверенность 94% | громкость 0.132 | срабатывание #1
🔔 [00:23] yes    | уверенность 88% | громкость 0.097 | срабатывание #2
```  
По Ctrl+C выводится статистика сессии, всё пишется в live_detect_log.txt.
Пороги (ENERGY_THRESHOLD, CONFIDENCE_THRESHOLD, COOLDOWN_SEC) настраиваются
в шапке live_detect.py.

# Конфигурация — code/config.py
|Параметр|По умолчанию|Описание|
|--------|------------|--------|
|audio.sample_rate|16000|Частота дискретизации|
audio.max_duration_sec|1.0|Длина окна спектрограммы
model.num_classes|10|Количество классов
model.use_attention|False|Attention-pooling после LSTM
training.batch_size|512|Размер батча
training.num_epochs|200|Максимум эпох (работает early stopping)
training.patience|20|Терпение early stopping
training.use_augmentation|True|SpecAugment на train
training.use_mixup|True|Mixup augmentation


## Результаты
|Метрики качества:||
|-----------------|-------------|
  Accuracy    : |0.9687 (96.9%)
  Precision   : |0.9688 (96.9%)
  Recall      : |0.9687 (96.9%)
  F1          : |0.9687 (96.9%)

  Confusion matrix сохраняется в data/processed/confusion_matrix.png, история обучения — в checkpoints/history.csv.