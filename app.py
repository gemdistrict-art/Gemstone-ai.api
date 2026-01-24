import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Načti model při startu
model_path = os.path.join(os.path.dirname(__file__), 'sample_model.h5')
model = load_model(model_path)

# Načti názvy tříd
with open('classes.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img = img.resize((224, 224))  # Musí odpovídat trénovacímu vstupu
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        pred = model.predict(img_array)
        predicted_class = class_names[int(np.argmax(pred))]
        confidence = float(np.max(pred))

        return jsonify({
            'gemstone': predicted_class,
            'confidence': round(confidence, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Health check pro Render
@app.route('/health')
def health():
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(debug=True)
