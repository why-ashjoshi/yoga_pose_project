import tensorflow as tf
from data_preprocessing import create_data_generators
from model_building import build_model

def train_model():
    train_dir = 'data/train'
    validation_dir = 'data/validation'
    
    train_generator, validation_generator = create_data_generators(train_dir, validation_dir)
    
    num_classes = len(train_generator.class_indices)
    model = build_model(num_classes)
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=10,
        validation_data=validation_generator
    )
    
    model.save('models/yoga_pose_model.h5')

if __name__ == "__main__":
    train_model()
