import tensorflow as tf
import os

os.makedirs("results", exist_ok=True)

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(e)

from sklearn.model_selection import train_test_split
from models import DenseNet_model, Inception_model, ResNet_model
from dataset_loader import load_dataset

X,y = load_dataset("dataset2","lanczos")

X_train,X_temp,y_train,y_temp = train_test_split(X,y,test_size=0.25,random_state=42)

X_val,X_test,y_val,y_test = train_test_split(X_temp,y_temp,test_size=0.63)

model = DenseNet_model()

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val,y_val),
    epochs=10,
    batch_size=32
)

model.save("results/densenet_model.h5")