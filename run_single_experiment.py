import sys
import pandas as pd
from dataset_loader import load_dataset
from models import DenseNet_model, Inception_model, ResNet_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import gc

# -------------------------------
# INPUT ARGUMENTS
# -------------------------------
# Example:
# python run_single_experiment.py lanczos DenseNet201

interp = sys.argv[1]
model_name = sys.argv[2]

# -------------------------------
# MODEL SELECTION
# -------------------------------
models = {
    "InceptionV3": Inception_model,
    "ResNet50V2": ResNet_model,
    "DenseNet201": DenseNet_model
}

print(f"\nRunning: {interp} + {model_name}")

# -------------------------------
# LOAD DATA
# -------------------------------
train_data, val_data, test_data = load_dataset("dataset_processed", interp)

train_data.reset()
val_data.reset()

# -------------------------------
# CLEAR MEMORY
# -------------------------------
K.clear_session()
gc.collect()

# -------------------------------
# BUILD MODEL
# -------------------------------
model = models[model_name]()

# -------------------------------
# EARLY STOPPING
# -------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# -------------------------------
# TRAIN
# -------------------------------
model.fit(
    train_data,
    validation_data=val_data,
    epochs=20,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------
# EVALUATE
# -------------------------------
loss, acc = model.evaluate(
    test_data,
    steps=test_data.samples // test_data.batch_size
)

print(f"\nFinal Accuracy: {acc:.4f}")

# -------------------------------
# SAVE RESULT
# -------------------------------
df = pd.DataFrame([{
    "Interpolation": interp,
    "Model": model_name,
    "Accuracy": round(acc, 4)
}])

file_path = "results/table6_results.csv"

try:
    old_df = pd.read_csv(file_path)
    df = pd.concat([old_df, df], ignore_index=True)
except:
    pass

df.to_csv(file_path, index=False)

print("\nResult saved ")


# -------------------------------
# CLEANUP
# -------------------------------
del model
gc.collect()