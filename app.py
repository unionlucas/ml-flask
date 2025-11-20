from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# ---------------------------------
# 1. Flask App erstellen
# ---------------------------------
app = Flask(__name__)

# ---------------------------------
# 2. Modell und Feature-Namen laden
# ---------------------------------
model = joblib.load("model.pkl")
try: 
    feature_names = joblib.load("feature_names.pkl")
except Exception:
    feature_names = None

CLASS_LABELS = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# ---------------------------------
# 3. Startseite 
# ---------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "features" not in data:
        return jsonify({"error": "Bitte JSON mit Key 'featues' senden"}), 400
    
    features = data["features"]

    if len(features) != 4:
        return jsonify({"error": "Es werden genau 4 Features erwartet"}), 400
    
    X = np.array(features).reshape(1, -1)

    pred = int(model.predict(X)[0]) # 0, 1 oder 2
    proba = model.predict_proba(X)[0].tolist()

    class_label = CLASS_LABELS.get(pred, f"Unbekannte Klasse {pred}")

    s = sum(features)
    return jsonify({
        "input": features,
        "prediction": pred,
        "class_label": class_label,
        "probabilities": proba
    })


if __name__ == "__main__":
    app.run(debug=True)
