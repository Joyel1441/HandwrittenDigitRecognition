from flask import Flask,render_template,request
import os
import numpy as np
import tensorflow
import keras
import PIL
from PIL import Image, ImageOps
import keras.preprocessing.image as img
from Preprocess import Preprocess

app = Flask(__name__)

@app.route("/")
def index():
   return render_template("index.html")

@app.route("/form",methods=["POST"])
def form():
    file_path = "" 
    # load the model
    cnn = keras.models.load_model('dr.h5')
    try:
       # get the image
       if request.method == "POST":
         image = request.files['image']
         img_name = image.filename
         file_path = os.path.join('./static/uploaded_images', img_name)
         image.save(file_path)
         # preprocess the image to make it similar
         # to training data
         a = Preprocess()
         a.preprocess(file_path)
         # image is converted to grayscale
         # and then to numpy array
         image = Image.open('./static/uploaded_images/preprocessed.jpeg') 
         image = ImageOps.grayscale(image)
         img_arr = img.img_to_array(image)
         img_arr = img_arr.astype("float32")
         img_arr = img_arr / 255.0
         img_arr = np.expand_dims(img_arr,axis = 0)
         # prediction
         predict = cnn.predict(img_arr)
         pred = np.argmax(predict[0])
         os.remove(file_path)
         os.remove('./static/uploaded_images/preprocessed.jpeg')
         return render_template("index.html",image_name=pred)
       else:
        return render_template("index.html",image_name="None")  
    except:
        return render_template("index.html",image_name="No proper image file selected")   
if __name__ == "__main__":
  app.run() 
  
