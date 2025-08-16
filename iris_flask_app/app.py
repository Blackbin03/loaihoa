from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model
model = pickle.load(open("iris_model.pkl", "rb"))
class_names = ["Setosa", "Versicolor", "Virginica"]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probabilities = None

    if request.method == "POST":
        try:
            sepal_length = float(request.form["sepal_length"])
            sepal_width = float(request.form["sepal_width"])
            petal_length = float(request.form["petal_length"])
            petal_width = float(request.form["petal_width"])

            features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
            prediction_index = model.predict(features)[0]
            prediction = class_names[prediction_index]

            probs = model.predict_proba(features)[0]
            probabilities = {class_names[i]: probs[i] for i in range(len(class_names))}
        except Exception as e:
            prediction = f"Lỗi nhập dữ liệu: {e}"

    return render_template("index.html", prediction=prediction, probabilities=probabilities)

if __name__ == "__main__":
    app.run(debug=True)
