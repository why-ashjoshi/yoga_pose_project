import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf

def make_inference(img_path):
    model = tf.keras.models.load_model('models/yoga_pose_model.h5')
    
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis

    predictions = model.predict(img_array)
    print(predictions)
    # Optionally, map predictions to class names if you have them

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1]  # Pass the image path as a command line argument
    make_inference(img_path)
