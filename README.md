# BCCD — Binary Classifier Cat-Dog 

Нейронная сеть для классификации звуков животных (кошек и собак) на основе анализа аудиосигналов.


Проект реализует гибридную архитектуру CNN + LSTM для классификации аудиофайлов. Звуковые файлы преобразуются в мел-спектрограммы, которые затем анализируются нейронной сетью.


1. Клонируйте репозиторий или скачайте проект

2. Создайте виртуальное окружение:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Установите зависимости:
```bash
pip install -r requirements
```
    Для работы CUDA (укажите подходящую вам версию CUDA, у меня 12.1)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

```    

## Использование

1. Предобработка данных

Поместите WAV файлы в папки `data/raw/cat/` и `data/raw/dog/`, затем запустите:

```bash
python code/preprocessing.py
```

2. Разбиение данных

```bash
python code/split_data.py
```

3. Обучение модели

```bash
python code/train.py
```

4. Оценка качества

```bash
python code/evaluate.py
```

5. Предсказание
    Для одного файла:
```bash
python predict.py audio.wav
```
    Для папки с файлами:
```bash
python predict.py audio_folder/
```