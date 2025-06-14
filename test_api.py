import requests

data = {
    "age": 50,
    "bmi": 25,
    "bp": 80,
    "s1": 100,
    "s2": 90,
    "s3": 85,
    "s4": 4.5,
    "s5": 4.2,
    "s6": 90
}

response = requests.post("http://127.0.0.1:5000/predict", json=data)

print("Prediction:", response.json())