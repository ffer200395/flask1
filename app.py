from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Cargar el modelo preentrenado
model = pickle.load(open('models/iris_model.pkl', "rb"))

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/iris")
def iris():
    return render_template('iris.html')

@app.route('/predict',methods = ['POST', 'GET'])
def predict():
  if request.method == 'POST':
      sl = request.form['sl']
      sw = request.form['sw']
      pl = request.form['pl']
      pw = request.form['pw']
      data = pd.DataFrame({
                    'sepal length (cm)': [float(sl)],
                    'sepal width (cm)': [float(sw)],
                    'petal length (cm)': [float(pl)],
                    'petal width (cm)': [float(pw)]
                })
                
      prediction = model.predict(data)
      return render_template("pred.html", value=prediction)
    
if __name__ == "__main__":
    app.run()
