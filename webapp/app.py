from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('models/yoga_pose_model.h5')
class_labels = ['vriksasana', 'tadasana', 'adho mukha svanasana']  # Update with your actual class labels

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    img_data = data['image'].split(',')[1]
    img_data = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_data))
    img = img.convert('RGB')  # Ensure image is in RGB format
    
    img = img.resize((150, 150))  # Resize to match model input
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array = img_array / 255.0  # Normalize if needed
    
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]
    
    return jsonify({'class': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
