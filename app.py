import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Load or create and train model
def load_or_create_model():
    try:
        model = load_model('pomegranate_model.h5')
        scaler = StandardScaler()
        scaler.mean_ = np.load('scaler_mean.npy')
        scaler.scale_ = np.load('scaler_scale.npy')
    except:
        model, scaler = create_and_train_model()
    return model, scaler

def create_and_train_model():
    np.random.seed(42)
    n_samples = 1000

    length = np.random.uniform(70, 100, n_samples)
    width = np.random.uniform(70, 100, n_samples)
    thickness = np.random.uniform(70, 100, n_samples)

    volume = (np.pi / 6) * length * width * thickness
    lateral_surface_area = np.pi * ((length * width + length * thickness + width * thickness) / 3)
    sphericity = (length * width * thickness)**(1/3) / length

    df = pd.DataFrame({
        'length': length,
        'width': width,
        'thickness': thickness,
        'volume': volume,
        'lateral_surface_area': lateral_surface_area,
        'sphericity': sphericity
    })

    X = df[['length', 'width', 'thickness']]
    y = df[['volume', 'lateral_surface_area', 'sphericity']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(3,)),
        Dense(32, activation='relu'),
        Dense(3)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.fit(X_scaled, y, epochs=100, batch_size=32, verbose=0)

    model.save('pomegranate_model.h5')
    np.save('scaler_mean.npy', scaler.mean_)
    np.save('scaler_scale.npy', scaler.scale_)

    return model, scaler

model, scaler = load_or_create_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        length = float(data['length'])
        width = float(data['width'])
        thickness = float(data['thickness'])

        input_data = np.array([[length, width, thickness]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)

        volume = float(prediction[0][0])
        lateral_surface_area = float(prediction[0][1])
        sphericity = float(prediction[0][2])

        # Calculate weight
        weight = volume / 1000 * 1.06  # Convert mm³ to cm³ and use density

        result = {
            'volume': volume,
            'lateral_surface_area': lateral_surface_area,
            'sphericity': sphericity,
            'weight': weight
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)