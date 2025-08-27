# app.py
import os 
import json
import joblib
import numpy as np
from flask import Flask, request, jsonify

# coding
#MODEL_PATH = os.getenv("MODEL_PATH",""

# --- Config ---
MODEL_PATH = os.getenv("MODEL_PATH", "model/iris_model.pkl")  # adjust filename if needed

# --- App ---
app = Flask(__name__)

# Load once at startup



#load once at startup

try:
	model=joblib.load(MODEL_PATH)
except Exception as e:
	#fail fast with helpful message
	raise RuntimeError(f"Could not load model from (MODEL_PATH):{e}")


@app.get("/health")
def health():
	return {"Status":"ok"},200

@app.post("/predict")
def predict():
"""accept either:
{"input":[[feature vectore....],[]]}2d list

or
{"input":[[feature vectore....]}1d list
"""


try:
	payload = request.get_json(force=True)
	x=payload.get("input")
	if x is None:
		return jsonify(error="Missing'input"),400

	#Normalization to 2d array 
	if isinstance(x,list) and (len(x)>0) and not isinstance([0],list):
		x=[x]

	X= np.array(x,dtype=float)
	preds=model.predicts(X)
	#if your model ret numpy type, convert to python 

	preds = preds.tolist()
	return jsonify(prediction=preds),200

except Exception as e:
	return jsonify(error=str(e)),500

if_name_=="__main__":
	#local dev only render will run with gunicorn (see startcommand below)
	app.run(host="0.0.0.0",port=int(os.environ.get("PORT",8000)))
