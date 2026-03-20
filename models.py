from tensorflow.keras.applications import DenseNet201, InceptionV3, ResNet50V2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -------------------------------
# COMMON BUILD FUNCTION (FIXED)
# -------------------------------
def build_model(base_model):

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output

    # 🔥 IMPORTANT FIX (NO FLATTEN)
    x = GlobalAveragePooling2D()(x)

    x = Dense(512, activation="swish")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# -------------------------------
# TABLE 6 MODELS
# -------------------------------
def DenseNet_model():
    base = DenseNet201(weights="imagenet", include_top=False, input_shape=(299,299,3))
    return build_model(base)


def Inception_model():
    base = InceptionV3(weights="imagenet", include_top=False, input_shape=(299,299,3))
    return build_model(base)


def ResNet_model():
    base = ResNet50V2(weights="imagenet", include_top=False, input_shape=(299,299,3))
    return build_model(base)


# -------------------------------
# TABLE 8 MODELS
# -------------------------------
def DenseNet_baseline():
    base = DenseNet201(weights="imagenet", include_top=False, input_shape=(299,299,3))

    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def DenseNet_proposed():
    base = DenseNet201(weights="imagenet", include_top=False, input_shape=(299,299,3))

    for layer in base.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base.output)
    x = Dense(512, activation="swish")(x)
    x = Dropout(0.2)(x)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model

def DenseNet_finetune():
    base = DenseNet201(weights="imagenet", include_top=False, input_shape=(299,299,3))

    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers[-30:]:
        layer.trainable = True

    x = GlobalAveragePooling2D()(base.output)
    output = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=base.input, outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model