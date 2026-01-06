import sqlite3
import random
import os
import csv
import base64
from PIL import Image
from io import BytesIO
from flask import Flask, render_template, request, send_from_directory, session, url_for, redirect
from src.utils.all_utils import read_yaml, create_directory
import os
from PIL import Image
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm

app = Flask(__name__)
app.secret_key = '\xf0?a\x9a\\\xff\xd4;\x0c\xcbHi'

def base64_to_image(base64_data):
    base64_data = base64_data.split(",")[-1]
    image_data = base64.b64decode(base64_data)
    image_stream = BytesIO(image_data)
    image = Image.open(image_stream)
    return image

connection = sqlite3.connect('user_data.db')
cursor = connection.cursor()

command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
cursor.execute(command)

def total_iems():
    return len(os.listdir("static/collections"))

config = read_yaml('config/config.yaml')
params = read_yaml('params.yaml')

artifacts = config['artifacts']
artifacts_dir = artifacts['artifacts_dir']

#upload
upload_image_dir = artifacts['upload_image_dir']
uploadn_path = os.path.join(artifacts_dir, upload_image_dir)

# pickle_format_data_dir
pickle_format_data_dir = artifacts['pickle_format_data_dir']
img_pickle_file_name = artifacts['img_pickle_file_name']

raw_local_dir_path = os.path.join(artifacts_dir, pickle_format_data_dir)
pickle_file = os.path.join(raw_local_dir_path, img_pickle_file_name)

#Feature path
feature_extraction_dir = artifacts['feature_extraction_dir']
extracted_features_name = artifacts['extracted_features_name']

feature_extraction_path = os.path.join(artifacts_dir, feature_extraction_dir)
features_name = os.path.join(feature_extraction_path, extracted_features_name)

#params_path
weight = params['base']['weights']
include_tops = params['base']['include_top']

#loading
feature_list = np.array(pickle.load(open(features_name,'rb')))
filenames = pickle.load(open(pickle_file,'rb'))


#model
model = ResNet50(weights= weight,include_top=include_tops,input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

def save_uploaded_file(uploaded_file):
    try:
        create_directory(dirs=[uploadn_path])
        with open(os.path.join(uploadn_path,uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


def recommend(features,feature_list):
    neighbors = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)

    distances, indices = neighbors.kneighbors([features])

    return indices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    items = os.listdir('static/data')
    files = random.sample(items, 12)
    return render_template('home.html', ti = total_iems(), files=files)

@app.route('/recommendation')
def recommendation():
    return render_template('recommend.html', ti = total_iems())

@app.route('/trial/<img>')
def trial(img):
    session['filename'] = 'static/data/'+img
    return render_template('trialroom.html', ti = total_iems())

@app.route('/cart')
def cart():
    List = []
    prices = []
    for im in os.listdir("static/collections"):
        f = open("prices.csv", "r")
        reader = csv.reader(f)
        File = im.split('.')[0]
        for i in reader:
            if File in i:
                prices.append(i[1])
                List.append("http://127.0.0.1:5000/static/collections/"+im)
    return render_template('cart.html', ti = total_iems(), List = List, prices = prices, n = len(List))

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return redirect(url_for('home'))

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/imagetest', methods=['GET', 'POST'])
def imagetest():
    if request.method == 'POST':
        fileName = session['filename']
        fileName1 = request.form['img2']
        File = fileName.split('.')[0]
        File = File.replace('static/data/', '')
        fileName1 = "static/test_img/"+fileName1

        os.system(f"python detection.py --input_image {fileName1} --input_cloth {fileName}")

        f = open("prices.csv", "r")
        reader = csv.reader(f)
        for i in reader:
            if File in i:
                name = i[1]
        import random
        random_range = random.randint(80, 90)
        return render_template('trialroom.html', ti = total_iems(), dress=fileName, price=name, image=fileName1, output="static/result/"+os.listdir("static/result")[0],random_range=random_range)

    return render_template('recommend.html' , ti = total_iems())

@app.route('/livetest', methods=['GET', 'POST'])
def livetest():
    if request.method == 'POST':
        fileName = session['filename']
        filedata = request.form['img2']

        dlist = os.listdir('static/testpicture')
        for item in dlist:
            os.remove("static/testpicture/"+item)
        
        name1 = str(random.randint(1000, 9999))
        result_image = base64_to_image(filedata)
        result_image.save('static/testpicture/'+name1+'.png')

        File = fileName.split('.')[0]
        File = File.replace('static/data/', '')
        fileName = fileName
        fileName1 = 'static/testpicture/'+name1+'.png'

        os.system(f"python detection.py --input_image {fileName1} --input_cloth {fileName}")

        f = open("prices.csv", "r")
        reader = csv.reader(f)
        for i in reader:
            if File in i:
                name = i[1]

        return render_template('trialroom.html', ti = total_iems(),  dress=fileName, price=name, image=fileName1, output="static/result/"+os.listdir("static/result")[0])

    return render_template('recommend.html', ti = total_iems())

##@app.route('/uploads/<filename>')
##def uploaded_file(filename):
##    return send_from_directory(uploadn_path, filename)

@app.route('/choose', methods=['POST'])
def choose():
    Type = request.form['Type']
    session['type'] = Type
    if Type == 'small':
        images = []
        names = []
        for img in os.listdir('static/artifacts/SMALL'):
            images.append('static/artifacts/SMALL/'+img)
            names.append(img)
        return render_template('recommend.html', names=names, images=images, Type=Type, n=len(images))
    if Type == 'medium':
        images = []
        names = []
        for img in os.listdir('static/artifacts/MEDIUM'):
            images.append('static/artifacts/MEDIUM/'+img)
            names.append(img)
        return render_template('recommend.html', names=names, images=images, Type=Type, n=len(images))
    if Type == 'large':
        images = []
        names = []
        for img in os.listdir('static/artifacts/LARGE'):
            images.append('static/artifacts/LARGE/'+img)
            names.append(img)
        return render_template('recommend.html', names=names, images=images, Type=Type, n=len(images))
    if Type == 'x-large':
        images = []
        names = []
        for img in os.listdir('static/artifacts/EXTRA_LARGE'):
            images.append('static/artifacts/EXTRA_LARGE/'+img)
            names.append(img)
        return render_template('recommend.html', names=names, images=images, Type=Type, n=len(images))
    return render_template('recommend.html')

@app.route('/upload/<name1>')
def upload(name1):
    print(name1)
    if session['type'] == 'small':
        filename = 'SMALL\\'+name1
    if session['type'] == 'medium':
        filename = 'MEDIUM\\'+name1
    if session['type'] == 'large':
        filename = 'LARGE\\'+name1
    if session['type'] == 'x-large':
        filename = 'EXTRA_LARGE\\'+name1
        
    features = feature_extraction(os.path.join(artifacts_dir, filename), model)
    indices = recommend(features, feature_list)
    result = []
    for i in filenames:
        result.append(i.replace("data\\", ''))
    return render_template('recommend.html', ti = total_iems(), filenames=result, indices=indices[0])

##        
@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
