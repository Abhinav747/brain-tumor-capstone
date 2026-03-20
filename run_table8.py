import pandas as pd
import numpy as np
from dataset_loader import load_dataset
from models import DenseNet_baseline, DenseNet_finetune, DenseNet_proposed
from metrics import compute_metrics
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import gc

interp = "lanczos"

results = []

models = {
    "Baseline DenseNet201": DenseNet_baseline,
    "Baseline + Fine-tuning": DenseNet_finetune,
    "Proposed DenseNet201": DenseNet_proposed,
}

for name, model_fn in models.items():

    print(f"\nRunning {name}")

    K.clear_session()
    gc.collect()

    train_data, val_data, test_data = load_dataset("dataset_processed", interp)

    model = model_fn()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    model.fit(
        train_data,
        validation_data=val_data,
        epochs=10,
        callbacks=[early_stop],
        verbose=1
    )

    # Predictions
    preds = model.predict(test_data)
    preds = (preds > 0.5).astype(int).flatten()

    y_true = test_data.classes

    # Metrics
    metrics = compute_metrics(y_true, preds)

    loss, acc = model.evaluate(test_data)

    results.append({
        "Model": name,
        "Accuracy": round(acc,4),
        "Precision": round(metrics["precision"],4),
        "Recall (TPR)": round(metrics["recall"],4),
        "Specificity": round(metrics["tnr"],4),
        "FPR": round(metrics["fpr"],4),
        "FNR": round(metrics["fnr"],4),
        "F1": round(metrics["f1"],4),
        "F2": round(metrics["f2"],4),
        "Jaccard": round(metrics["jaccard"],4),
        "Hamming": round(metrics["hamming"],4),
        "NPV": round(metrics["tnr"],4),
        "Loss": round(loss,4)
    })

    del model
    gc.collect()

df = pd.DataFrame(results)

print("\nTable 8 Results:\n")
print(df)

df.to_csv("results/table8_results.csv", index=False)