from pathlib import Path

import pandas as pd
from config import cfg
from sklearn.model_selection import train_test_split


def split_dataset(
    processed_dir=None,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):
    """
    Стратифицированное разбиение датасета на train/val/test.
    """
    processed_dir = processed_dir or cfg.paths.processed_dir
    processed_path = Path(processed_dir)
    metadata_path = processed_path / 'metadata.csv'
    df = pd.read_csv(metadata_path)

    print("=" * 50)
    print("РАЗБИЕНИЕ ДАТАСЕТА")
    print("=" * 50)
    print(f"Всего файлов: {len(df)}")
    print(f"\nРаспределение классов:")
    for cls, cnt in df['class'].value_counts().sort_index().items():
        print(f"  {cls}: {cnt} ({cnt / len(df) * 100:.1f}%)")

    # Проверка малых классов
    class_counts = df['class'].value_counts()
    too_small = class_counts[class_counts < 3]
    if not too_small.empty:
        print(f"\n⚠ ВНИМАНИЕ: классы с <3 примерами не смогут "
              f"быть представлены во всех сплитах:")
        for cls, cnt in too_small.items():
            print(f"  {cls}: {cnt}")

    # (train + val) vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size,
        random_state=random_state, stratify=df['label'],
    )

    # train vs val
    val_ratio = val_size / (train_size + val_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio,
        random_state=random_state, stratify=train_val_df['label'],
    )

    print(f"\nРазбиение:")
    print(f"  Train: {len(train_df)} файлов ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"  Val:   {len(val_df)} файлов ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"  Test:  {len(test_df)} файлов ({len(test_df) / len(df) * 100:.1f}%)")

    def print_class_distribution(name, subset_df):
        print(f"\n  {name}:")
        total = len(subset_df)
        for cls, cnt in subset_df['class'].value_counts().sort_index().items():
            print(f"    {cls}: {cnt} ({cnt / total * 100:.1f}%)")

    print(f"\nБаланс классов:")
    print_class_distribution("Train", train_df)
    print_class_distribution("Validation", val_df)
    print_class_distribution("Test", test_df)

    train_df.to_csv(processed_path / 'train.csv', index=False)
    val_df.to_csv(processed_path / 'val.csv', index=False)
    test_df.to_csv(processed_path / 'test.csv', index=False)

    print(f"\n✅ Файлы сохранены:")
    print(f"  {processed_path / 'train.csv'}")
    print(f"  {processed_path / 'val.csv'}")
    print(f"  {processed_path / 'test.csv'}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    split_dataset()