#app.py
import os
import json
from flask import Flask, request, jsonify
import joblib
import numpy as np

#--config--
MODEL_PATH = os.getenv('MODEL_PATH', 'model/iris_model.pkl')

#--App
app = Flask(__name__)


#load one at startuo
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    #fail fast with a helpful message
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

@app.get('/health')
def health():
    return {"status": "ok"}, 200

@app.post('/predict')
def predict():
    '''
    accepts either:
    {"input " : [[...feature vectors...],[...]]} #2D list
    or
    {"input" : [...feature vector...]} #1D list
    '''
    try:
        payload = request.get_json(force=True)
        x = payload.get('input')
        if x is None:
            return jsonify(error="Missing 'input'"), 400
        # normalize to 2D array
        if isinstance(x, list) and (len(x) > 0) and isinstance(x[0], list):
            X = x
        else:
            X = [x]
        X = np.array(X, dtype=float)
        preds = model.predict(X)
        preds = preds.tolist()
        return jsonify(predictions=preds), 200
    except Exception as e:
        return jsonify(error=str(e)), 500   
if _name_ == '_main_':
    #app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)))
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 8000)))

