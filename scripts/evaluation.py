import tensorflow as tf
from data_preprocessing import create_data_generators

def evaluate_model():
    test_dir = 'data/test'  # Ensure this directory exists and contains your test data

    # Load the test data using a separate function
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical'
    )

    model = tf.keras.models.load_model('models/yoga_pose_model.h5')

    test_loss, test_acc = model.evaluate(test_generator)
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_acc}')

if __name__ == "__main__":
    evaluate_model()
