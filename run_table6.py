import pandas as pd
from dataset_loader import load_dataset
from models import DenseNet_model, Inception_model, ResNet_model
from tensorflow.keras import backend as K
import gc

interpolations = ["nearest","bilinear","bicubic","lanczos"]

results = []

models = {
    "InceptionV3": Inception_model,
    "ResNet50V2": ResNet_model,
    "DenseNet201": DenseNet_model
}

for interp in interpolations:

    print(f"\nRunning interpolation: {interp}")

    train_data, val_data, test_data = load_dataset("dataset_processed", interp)

    for name, model_fn in models.items():

        print(f"Training {name}")

        K.clear_session()
        gc.collect()

        model = model_fn()

        model.fit(
            train_data,
            validation_data=val_data,
            epochs=2,
            verbose=1
        )

        loss, acc = model.evaluate(test_data)

        results.append({
            "Interpolation": interp,
            "Model": name,
            "Accuracy": round(acc,4)
        })

        del model
        gc.collect()

df = pd.DataFrame(results)

print("\nFinal Table 6 Results:\n")
print(df)

df.to_csv("results/table6_results.csv", index=False)