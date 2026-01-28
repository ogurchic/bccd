#скрипт для извлечения категорий звуков с репозитория https://github.com/karolpiczak/ESC-50/tree/master?tab=readme-ov-file#repository-content

import pandas as pd
import shutil
from pathlib import Path

meta = pd.read_csv('data/ESC-50/meta/esc50.csv')

# можно поставить необходжимые метки звуков
dogs = meta[meta['category'] == 'dog']['filename'].tolist()
cats = meta[meta['category'] == 'cat']['filename'].tolist()


for f in dogs:
    shutil.copy(f'data/ESC-50/audio/{f}', f'data/raw/dog/{f}')

for f in cats:
    shutil.copy(f'data/ESC-50/audio/{f}', f'data/raw/cat/{f}')