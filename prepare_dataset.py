import os
import cv2
import random
import shutil
from sklearn.model_selection import train_test_split

IMG_SIZE = 299

INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4
}

SOURCE = "dataset2"
DEST = "dataset_processed"

def preprocess_and_split(method):

    print(f"\nProcessing {method}")

    temp_dir = f"{DEST}/{method}/temp"

    for label in ["yes","no"]:
        os.makedirs(f"{temp_dir}/{label}", exist_ok=True)

        images = os.listdir(f"{SOURCE}/{label}")
        images = random.sample(images, 600)   # same as paper

        for img in images:
            img_path = f"{SOURCE}/{label}/{img}"
            image = cv2.imread(img_path)

            if image is None:
                continue

            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE),
                               interpolation=INTERPOLATIONS[method])

            cv2.imwrite(f"{temp_dir}/{label}/{img}", image)

    # SPLIT (same as paper)
    for label in ["yes","no"]:

        images = os.listdir(f"{temp_dir}/{label}")

        train, temp = train_test_split(images, test_size=0.25, random_state=42)
        val, test = train_test_split(temp, test_size=0.63)

        for split, split_data in zip(["train","val","test"], [train,val,test]):

            out_dir = f"{DEST}/{method}/{split}/{label}"
            os.makedirs(out_dir, exist_ok=True)

            for img in split_data:
                shutil.copy(f"{temp_dir}/{label}/{img}", out_dir)

    shutil.rmtree(temp_dir)


for method in INTERPOLATIONS:
    preprocess_and_split(method)

print("\n Dataset prepared for Table 6")