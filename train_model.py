import joblib
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# 1. Datensatz laden
iris = load_iris()
X = iris.data
y = iris.target

# 2. Trainings-/Testdaten erzeugen
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,      # 20% als Testdaten
    random_state=42,    # Reproduzierbarkeit
    stratify=y          # Ähnliche Klassenverteilung in Train/Test
)

# 3. Modell trainieren
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Genauigkeit ausgeben
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model-Accuracy (3 Klassen): {acc:.3f}")

# 5. Modell und Feature-Namen speichern
joblib.dump(model, "model.pkl")
joblib.dump(iris.feature_names, "feature_names.pkl"),