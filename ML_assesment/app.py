from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import os

def load_model():
    try:
        # Specify the absolute path to the ml_model folder
        model_path = os.path.join(os.path.dirname(os.path.abspath(_file_)), 'F://ML_assesment//Model')
        # Load the model
        model = tf.keras.models.load_model(model_path)
        # Check if the loaded object is a model
        if not isinstance(model, tf.keras.Model):
            raise ValueError("Loaded object is not a valid TensorFlow model.")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Load the model when your application starts
model = load_model()

def run_model_and_update_file(shared_value):
    try:
        # Generate new data for testing
        X_test = np.array([[shared_value]])
        
        # Check if the model is loaded successfully
        if model is not None:
            # Predict the output using the trained model
            y_pred = model.predict(X_test)
            
            # Check if the loaded model has the 'predict' method
            if not hasattr(model, 'predict'):
                raise AttributeError("Loaded model does not have a 'predict' method.")
            
            # Write the predicted value back to the file
            with open('shared_value.txt', 'w') as file:
                file.write(str(y_pred[0][0]))

            return y_pred[0][0]
        else:
            return None
    except Exception as e:
        print(f"Error running model: {e}")
        return None

app = Flask(_name_)
CORS(app)

@app.route('/home')
def home():
    return render_template('home.html')

# Define a route for the GET request
@app.route('/', methods=['GET', 'POST'])
def result():
    global shared_value  # Use global to modify the shared_value

    if request.method == 'GET':
        return jsonify({"message": f"current value is {shared_value}"})
      
    elif request.method == 'POST':
        data = request.json  # Assuming the data is sent as JSON
        if 'new_value' in data and isinstance(data['new_value'], int):
            # Update shared_value with the new integer value
            shared_value = data['new_value']
            
            # Save the updated shared_value to the file
            new_value = run_model_and_update_file(shared_value)
            
            if new_value is not None:
                return jsonify({"message": f"corresponding value: {new_value}"})
            else:
                return jsonify({"error": "Error running the model"}), 500
        else:
            return jsonify({"error": "Invalid data format"}), 400

if _name_ == '_main_':
    # Run the Flask app
    app.run(debug=True, port=3000)