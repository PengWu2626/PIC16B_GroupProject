from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
from flask_dropzone import Dropzone
import pandas as pd
import numpy as np
import tensorflow as tf


from src import face_detect
from src import dog_recommendation
from src import dog_classes
from src import dogtime_barcharts


# import plotly
# import plotly.express as px
# import json

app = Flask(__name__)
dropzone = Dropzone(app)

UPLOAD_FOLDER = 'static/uploads/'
DEST_FOLDER = 'static/faces/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
DRAG_UPLOAD_NAME=''

# 120 breed labels from training dataset "stanford dog dataset"
classname = dog_classes.CLASS_NAME

# Models
# cat_dog_model from homework blog post 5
cat_dog_model = tf.keras.models.load_model('static/models/blog_post_model4_logit2.h5')
# used transfer learning Xception
model2 = tf.keras.models.load_model('static/models/dogmodel2.h5')

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Dropzone , drag to upload
# https://flask-dropzone.readthedocs.io/en/latest/
# some dropzone examples
# https://github.com/greyli/flask-dropzone/tree/master/examples
basedir = os.path.abspath(os.path.dirname(__file__))
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'static/uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='image',
    DROPZONE_MAX_FILE_SIZE=20,
    DROPZONE_MAX_FILES=1,
    DROPZONE_IN_FORM=True,
    DROPZONE_UPLOAD_ON_CLICK=True,
    DROPZONE_UPLOAD_ACTION='drag_save',  # URL or endpoint
    DROPZONE_UPLOAD_BTN_ID='dragsubmit_btn',
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def dog_breed_prediction (path):
    # dimensions of our images
    IMG_HEIGHT = 299
    IMG_WIDTH = 299

    # predicting images
    img = tf.keras.preprocessing.image.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(img, axis=0)/255.

    # Get predicted probabilities for 120 class labels
    pred_classes = model2.predict(x, batch_size=32)
    # Display image being classified
    get = np.argsort(pred_classes)
    get=get[0]
    # get top three encoded label
    three_dog = get[-1:-4:-1]

    most_likely_list=[]
    most_likely_probability_list=[]

    for i in three_dog:
        # find associated dog from encoded label
        most_likely_list.append(classname[i])
        # probability for each breed
        most_likely_probability_list.append(((pred_classes[0][i])*100).round(4))

    return (most_likely_list, most_likely_probability_list)



def cat_or_dog(img_path):

    IMG_HEIGHT = 160
    IMG_WIDTH = 160

    # https://www.tensorflow.org/tutorials/images/transfer_learning
    img = tf.keras.utils.load_img(
        img_path, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img, 0) # Create a batch

    predictions = cat_dog_model.predict_on_batch(img).flatten() #logit
    probability_cat_dog = tf.nn.softmax(predictions)            # probability

    class_names=['cat', 'dog']

    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(probability_cat_dog)], 100 * np.max(probability_cat_dog)))
    
    # predicting class name
    y_label = class_names[np.argmax(probability_cat_dog)]
    y_confidence = (100 * np.max(probability_cat_dog)).round(3)

    return y_label, y_confidence


def top_three_images(most_likely_breeds_list):
    first_pic_path = 'static/dogImages'+'/' + most_likely_breeds_list[0]
    second_pic_path = 'static/dogImages'+'/' + most_likely_breeds_list[1]
    third_pic_path = 'static/dogImages'+'/' + most_likely_breeds_list[2]

    # randome pick 4
    first_img_path_list  = np.random.choice([os.path.join(first_pic_path, x) for x in os.listdir(first_pic_path)], size=4, replace=False)
    second_pic_path_list = np.random.choice([os.path.join(second_pic_path, x) for x in os.listdir(second_pic_path)], size=4, replace=False)
    third_pic_path_list  = np.random.choice([os.path.join(third_pic_path, x) for x in os.listdir(third_pic_path)], size=4, replace=False)
    
    pic_path_list=[first_img_path_list, second_pic_path_list, third_pic_path_list]
    most_likely_breeds_list = [(x.split('-',1)[1]).replace('_',' ').title() for x in most_likely_breeds_list]

    return most_likely_breeds_list, pic_path_list




@app.route('/intro')
def intro():
    return render_template('intro.html')

@app.route('/')
def index():
    print(os. getcwd())
    return render_template('index.html')

@app.route('/view/')
def view():
    return render_template('view.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/display/<filename>')
def display_image(filename):
    # face detect
    num = face_detect.faceDetector(filename, UPLOAD_FOLDER, DEST_FOLDER)
    if (num):
       return redirect(url_for('static', filename='faces/' + filename), code=301)
    return redirect(url_for('static', filename='uploads/' + filename), code=302)



#  the first way to upload an image, so there are many repeated codes between uploads methods
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part!')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading!')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # flash('The image has been uploaded successfully!')

        any_face=0
        any_face = face_detect.faceDetector(filename, UPLOAD_FOLDER, DEST_FOLDER)

        uploaded_image_path = (os.path.join(UPLOAD_FOLDER, file.filename))
        catordog, catordog_confidence = cat_or_dog(uploaded_image_path)

        most_likely_breeds_list, most_likely_probability_list = dog_breed_prediction(uploaded_image_path)
        most_likely_breeds_list, pic_path_list = top_three_images(most_likely_breeds_list=most_likely_breeds_list)

        results = pd.read_csv('static/dogtime.csv')
        dog_breed_all = results['breed'].unique()

        if (most_likely_breeds_list[0] in dog_breed_all):
            dog_breed_all = np.sort(dog_breed_all)

            graphJSON, dogpic = dogtime_barcharts.dogtime_plot(which_dog = most_likely_breeds_list[0])

            df = results[results['breed']== most_likely_breeds_list[0]]
            dog_characteristic_list= list(df['characteristic'])
            dog_stars_list = list(df['star'])
            dog_time_info =list(zip(dog_characteristic_list,dog_stars_list))

            return render_template('view.html', filename=filename, catordog=catordog,catordog_confidence=catordog_confidence,
             which_breed=most_likely_breeds_list[0], graphJSON=graphJSON,dogurl = dogpic,in_df = True, any_face=any_face,
            pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list,
            dog_time_info=dog_time_info)

        return render_template('view.html', filename=filename, catordog=catordog, catordog_confidence=catordog_confidence,
         which_breed=most_likely_breeds_list, in_df = False, any_face=any_face,
         pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    



@app.route('/dragsave', methods=['POST'])
def drag_save():
    global DRAG_UPLOAD_NAME

    for key, f in request.files.items():
        if key.startswith('file'):
            # save the dragged image path to global variable UPPATH
            DRAG_UPLOAD_NAME= f.filename
            f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
    return DRAG_UPLOAD_NAME

# the second way to upload an image, so there are many repeated codes
@app.route('/dragupload', methods=['GET','POST'])
def drag_upload():
    if request.method == 'GET':
        return render_template('drag_upload.html')
    else:
        global DRAG_UPLOAD_NAME

        uploaded_image_path =os.path.join(app.config['UPLOADED_PATH'], DRAG_UPLOAD_NAME)
        any_face = 0
        any_face = face_detect.faceDetector(DRAG_UPLOAD_NAME, UPLOAD_FOLDER, DEST_FOLDER)

        catordog, catordog_confidence = cat_or_dog(uploaded_image_path)

        most_likely_breeds_list, most_likely_probability_list = dog_breed_prediction(uploaded_image_path)
        most_likely_breeds_list, pic_path_list = top_three_images(most_likely_breeds_list=most_likely_breeds_list)
        

        results = pd.read_csv('static/dogtime.csv')
        dog_breed_all = results['breed'].unique()
        if (most_likely_breeds_list[0] in dog_breed_all):
            dog_breed_all = np.sort(dog_breed_all)
            graphJSON, dogpic = dogtime_barcharts.dogtime_plot(which_dog = most_likely_breeds_list[0])

            df = results[results['breed']== most_likely_breeds_list[0]]
            dog_characteristic_list= list(df['characteristic'])
            dog_stars_list = list(df['star'])
            dog_time_info =list(zip(dog_characteristic_list,dog_stars_list))

            return render_template('view.html', filename=DRAG_UPLOAD_NAME, catordog=catordog,catordog_confidence=catordog_confidence,
                which_breed=most_likely_breeds_list[0], graphJSON=graphJSON,dogurl = dogpic,in_df = True, any_face=any_face,
            pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list,
            dog_time_info=dog_time_info)

        return render_template('view.html', filename=DRAG_UPLOAD_NAME, catordog=catordog, catordog_confidence=catordog_confidence,
            which_breed=most_likely_breeds_list, in_df = False, any_face=any_face,
            pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list)




# the third way to upload an image from uploaded folder, so there are many repeated codes
@app.route('/gallery',methods=['GET','POST'])
def get_gallery():

    uploads_links = [x for x in os.listdir('static/uploads')if (x.split('.')[-1]).lower() in ALLOWED_EXTENSIONS ]

    if request.method == 'GET':
        return render_template('display_uploads.html', uploads_links = uploads_links )
    else:

        # extract the value of submitbutton from request
        user_clicked_image_name =request.form.get('submitbutton')
        user_clicked_image_path = (os.path.join(UPLOAD_FOLDER, user_clicked_image_name))

        any_face = 0
        any_face = face_detect.faceDetector(user_clicked_image_name, UPLOAD_FOLDER, DEST_FOLDER)

        catordog, catordog_confidence = cat_or_dog(user_clicked_image_path)

        most_likely_breeds_list, most_likely_probability_list = dog_breed_prediction(user_clicked_image_path)

        most_likely_breeds_list, pic_path_list = top_three_images(most_likely_breeds_list=most_likely_breeds_list)
        
        results = pd.read_csv('static/dogtime.csv')
        dog_breed_all = results['breed'].unique()
        if (most_likely_breeds_list[0] in dog_breed_all):
            dog_breed_all = np.sort(dog_breed_all)

            graphJSON, dogpic = dogtime_barcharts.dogtime_plot(which_dog = most_likely_breeds_list[0])
            
            df = results[results['breed']== most_likely_breeds_list[0]]
            dog_characteristic_list= list(df['characteristic'])
            dog_stars_list = list(df['star'])
            dog_time_info =list(zip(dog_characteristic_list,dog_stars_list))

            return render_template('view.html', filename=user_clicked_image_name, catordog=catordog,catordog_confidence=catordog_confidence,
             which_breed=most_likely_breeds_list[0], graphJSON=graphJSON,dogurl = dogpic, in_df = True, any_face=any_face,
            pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list,
            dog_time_info=dog_time_info)

        return render_template('view.html', filename=user_clicked_image_name, catordog=catordog, catordog_confidence=catordog_confidence,
         which_breed=most_likely_breeds_list, in_df = False, any_face=any_face,
         pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list)
   



@app.route('/doginfo/', methods=['GET', 'POST'])
def doginfo():
    results = pd.read_csv('static/dogtime.csv')
    dog_breed_all = results['breed'].unique()
    dog_breed_all = np.sort(dog_breed_all)

    if request.method == 'GET':
        return render_template('doginformation.html', dog_breed_all=dog_breed_all)
    else:
        select = request.form['dogs']
        print(select)
        which_dog = select

        graphJSON, dogpic = dogtime_barcharts.dogtime_plot(which_dog = which_dog)

        return render_template('doginformation.html', graphJSON=graphJSON, dog_breed_all=dog_breed_all, select=select, dogurl = dogpic)

def get_a_dog_image(name):
    results = pd.read_csv('static/dogtime.csv')
    df = results[results['breed']==name]
    dogpic = df['image_src'].unique()[0]
    return dogpic


@app.route('/findyourdog', methods=['GET', 'POST'])
def findyourdog():
    if request.method == 'GET':
        return render_template('findyourdog.html', dogmap= dog_recommendation.prepare_recommendation_df()[1])
    else:
        # get user selected characteristics
        list_26 = request.form.getlist('characteristic_slider')
        list_26= np.asarray(list_26)
        characteristics = dog_recommendation.make_characteristics_map(list_26)
        recommendation_list = dog_recommendation.find_dog_recommendation(**characteristics)
        find_any = bool(recommendation_list)

        dog_picture_lick =[]
        for i in recommendation_list:
            dog_picture_lick.append(get_a_dog_image(i))

        dog_recommendations = zip(recommendation_list, dog_picture_lick)

        return render_template('findyourdog.html', dogmap=dog_recommendation.prepare_recommendation_df()[1],recommendation_list=recommendation_list,
                 find_any=find_any, dog_recommendations=dog_recommendations)



if __name__ == "__main__":
    app.run()


# export FLASK_ENV=development; flask run