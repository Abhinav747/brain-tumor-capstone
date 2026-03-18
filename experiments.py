import pandas as pd
from dataset_loader import load_dataset
from models import DenseNet_model, Inception_model, ResNet_model
from sklearn.model_selection import train_test_split

interpolations = ["nearest","bilinear","bicubic","lanczos"]

results = []

for method in interpolations:

    print(f"\nRunning interpolation: {method}")

    X,y = load_dataset("dataset2",method)

    X_train,X_temp,y_train,y_temp = train_test_split(
        X,y,test_size=0.25,random_state=42)

    X_val,X_test,y_val,y_test = train_test_split(
        X_temp,y_temp,test_size=0.63)

    models = {
        "InceptionV3": Inception_model(),
        "ResNet50V2": ResNet_model(),
        "DenseNet201": DenseNet_model()
    }

    for name,model in models.items():

        print(f"Training {name}")

        model.fit(
            X_train,y_train,
            validation_data=(X_val,y_val),
            epochs=10,
            batch_size=32,
            verbose=1
        )

        loss,acc = model.evaluate(X_test,y_test)

        results.append({
            "Interpolation":method,
            "Model":name,
            "Accuracy":acc
        })

df = pd.DataFrame(results)

print("\nFinal Results:")
print(df)

df.to_csv("results/table6_results.csv",index=False)