import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Flask utils
from flask import Flask, request, render_template

from werkzeug.utils import secure_filename  

# Define a flask app
app = Flask(__name__)
# Example of dynamic path construction
directory_name = "models"
file_name = "modelres50.h5"

# Constructing a path dynamically
dynamic_path = os.path.join(directory_name, file_name)

# Getting the absolute path
absolute_path = os.path.abspath(dynamic_path)

print("Dynamic path:", dynamic_path)
print("Absolute path:", absolute_path)
mode = absolute_path
# Model saved with Keras model.save()
#mode = '/Users/deveshpundir/web-app-BDT/models/modelres50.h5'


model = load_model(mode)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(200,200)) 

    # Preprocessing the image
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')/255  #feature scaling
   
    preds = model.predict(img)
    pred = np.argmax(preds,axis = 1)
    return pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)
        os.remove(file_path)            #removes file from the server after prediction has been returned

        str0 = 'Glioma'
        str1 = 'Meningioma'
        str3 = 'pituitary'
        str2 = 'No Tumour'
        if pred[0] == 0:
            return str0
        elif pred[0] == 1:
            return str1
        elif pred[0]==3:
            return str3
        else:
            return str2
    return None

if __name__ == '__main__':
        app.run(debug=True, host="localhost", port=5000)    #debug is true for the development phase
    
