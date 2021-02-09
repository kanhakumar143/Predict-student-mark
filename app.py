from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the Random Forest CLassifier model
filename = 'model.pkl'
Lr = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        hour = float(request.form['Hour'])
        
        data = np.array([hour]).reshape(1, 1)
        
        my_prediction = Lr.predict(data)

        output = round(my_prediction[0], 2)
        
        return render_template('index.html', prediction_text='predicted mark of a student  = {}'.format(output))

if __name__ == '__main__':
	app.run(debug=True)