from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def load_dataset(base_path, interpolation):

    train_dir = os.path.join(base_path, interpolation, "train")
    val_dir   = os.path.join(base_path, interpolation, "val")
    test_dir  = os.path.join(base_path, interpolation, "test")

    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen   = ImageDataGenerator(rescale=1./255)
    test_gen  = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        train_dir,
        target_size=(299,299),
        batch_size=8,
        class_mode='binary'
    )

    val_data = val_gen.flow_from_directory(
        val_dir,
        target_size=(299,299),
        batch_size=8,
        class_mode='binary'
    )

    test_data = test_gen.flow_from_directory(
        test_dir,
        target_size=(299,299),
        batch_size=8,
        class_mode='binary',
        shuffle=False
    )

    return train_data, val_data, test_data