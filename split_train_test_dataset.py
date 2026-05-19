import os
import shutil
import random
from tqdm import tqdm


def split_dataset(
        source_dir='MIT-BIH_AD_128x128',
        output_dir='dataset_split',
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
):
    random.seed(seed)

    # Créer les répertoires train/val/test
    splits = ['train', 'val', 'test']
    classes = os.listdir(source_dir)

    for split in splits:
        for cls in classes:
            os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)


    for cls in tqdm(classes, desc="Processing classes"):
        cls_path = os.path.join(source_dir, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)

        n = len(files)
        train_end = int(train_ratio * n)
        val_end = train_end + int(val_ratio * n)

        train_files = files[:train_end]
        val_files = files[train_end:val_end]
        test_files = files[val_end:]

        for split, split_files in zip(splits, [train_files, val_files, test_files]):
            for f in split_files:
                src = os.path.join(cls_path, f)
                dst = os.path.join(output_dir, split, cls, f)
                shutil.copy2(src, dst)

    print("✅ Dataset splitting completed!")


# Exécuter
split_dataset()
