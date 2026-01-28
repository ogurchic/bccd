import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_dataset(processed_dir, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
       
    processed_path = Path(processed_dir)
    
    metadata_path = processed_path / 'metadata.csv'
    df = pd.read_csv(metadata_path)
    
    print("Информация о датасете:")
    print(f"  Всего файлов: {len(df)}")
    print(f"  Распределение классов:")
    print(df['class'].value_counts().to_string())
    
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df['label']
    )
    
    val_ratio = val_size / (train_size + val_size)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=random_state,
        stratify=train_val_df['label']
    )
    
    print(f"\nРазбиение данных:")
    print(f"  Train:      {len(train_df)} файлов ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Validation: {len(val_df)} файлов ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:       {len(test_df)} файлов ({len(test_df)/len(df)*100:.1f}%)")
    
    
    print(f"\nБаланс классов:")
    
    print(f"  Train:")
    for class_name, count in train_df['class'].value_counts().items():
        # :.1f — форматирование числа с 1 знаком после запятой
        print(f"    {class_name}: {count} ({count/len(train_df)*100:.1f}%)")
    
    print(f"  Validation:")
    for class_name, count in val_df['class'].value_counts().items():
        print(f"    {class_name}: {count} ({count/len(val_df)*100:.1f}%)")
    
    print(f"  Test:")
    for class_name, count in test_df['class'].value_counts().items():
        print(f"    {class_name}: {count} ({count/len(test_df)*100:.1f}%)")

    train_df.to_csv(processed_path / 'train.csv', index=False)
    val_df.to_csv(processed_path / 'val.csv', index=False)
    test_df.to_csv(processed_path / 'test.csv', index=False)
    
    print(f"\nФайлы сохранены:")
    print(f"  {processed_path / 'train.csv'}")
    print(f"  {processed_path / 'val.csv'}")
    print(f"  {processed_path / 'test.csv'}")
    
    return train_df, val_df, test_df



if __name__ == "__main__":  
    PROCESSED_DIR = "data/processed"
    
    print("Разбиение датасета на train/val/test...\n")
    
    train_df, val_df, test_df = split_dataset(
        processed_dir=PROCESSED_DIR,
        train_size=0.7, # 70% для тренировки 
        val_size=0.15,  # 15% для валидационных данных
        test_size=0.15  # 15% для теста
    )
    
