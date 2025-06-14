from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([
        data["age"], data["bmi"], data["bp"], data["s1"], data["s2"],
        data["s3"], data["s4"], data["s5"], data["s6"]
    ]).reshape(1, -1)
    
    prediction = model.predict(features)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True, port=5000)