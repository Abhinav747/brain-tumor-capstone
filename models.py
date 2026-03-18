from tensorflow.keras.applications import DenseNet201, InceptionV3, ResNet50V2
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_model(base_model):

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    x = Dense(1024, activation="swish")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def DenseNet_model():
    base = DenseNet201(weights="imagenet", include_top=False, input_shape=(299,299,3))
    return build_model(base)


def Inception_model():
    base = InceptionV3(weights="imagenet", include_top=False, input_shape=(299,299,3))
    return build_model(base)


def ResNet_model():
    base = ResNet50V2(weights="imagenet", include_top=False, input_shape=(299,299,3))
    return build_model(base)