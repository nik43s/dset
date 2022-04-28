!pip install numpy 
!pip install sklearn


import numpy as np
import tensorflow as tf
import pickle
import sklearn as skl


from flask import Flask, request, render_template

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


print("Версия TensorFlow:", tf.__version__)


app = Flask(__name__)


def get_prediction(person):
    pkl_filename = "saved_model.pkl"
    with open(pkl_filename, 'rb') as file:
        model = pickle.load(file)
    y_pred = model.predict(person)

    return f"Модуль упругости при растяжении {y_pred}"



@app.route('/')
def index():
    return "main"


@app.route('/predict/', methods=['post', 'get'])
def processing():
    message = ''
    if request.method == 'POST':
        person = request.form.get('username')

        person_parameters = person.split(" ")
        person = [float(param) for param in person_parameters]
        person = np.array([person])

        message = get_prediction(person)

    return render_template('login.html', message=message)


if __name__ == '__main__':
    app.run()
