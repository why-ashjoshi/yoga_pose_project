import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generators(train_dir, validation_dir=None, target_size=(150, 150), batch_size=32):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    validation_generator = None
    if validation_dir:
        validation_datagen = ImageDataGenerator(rescale=1./255)
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=target_size,
            batch_size=batch_size,
            class_mode='categorical'
        )

    return train_generator, validation_generator
