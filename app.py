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
model2 = tf.keras.models.load_model('static/models/model2.h5')

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
    """
    This function will get an image path and predict the top 3 most likely dog breed
    from 120 dog breeds in the Stanford dog dataset
    
    Parameters
    ----------
    path: string; a dog image path

    Return 
    ----------
    most_likely_list: list of strings; top 3 most likely dog breed name
    most_likely_probability_list: list of floats; probability for each 3 dog breed
    """
    # dimensions of our images
    IMG_HEIGHT = 299
    IMG_WIDTH = 299

    # Image pre-processing
    img = tf.keras.preprocessing.image.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)/255.

    # Get predicted probabilities for 120 class labels
    pred_classes = model2.predict(img, batch_size=32)
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
    """
    This function will get an image path and predict a cat or dog.

    Parameters
    ----------
    img_path: string; a dog image

    Return 
    ----------
    y_label: string; predicted result
    y_confidence: confidence score
    """
    IMG_HEIGHT = 160
    IMG_WIDTH = 160

    # Image pre-processing
    # https://www.tensorflow.org/tutorials/images/transfer_learning
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = tf.keras.utils.img_to_array(img)
    img = tf.expand_dims(img, 0) # Create a batch

    predictions = cat_dog_model.predict_on_batch(img).flatten()  # logit
    probability_cat_dog = tf.nn.softmax(predictions)             # probability

    class_names=['cat', 'dog']

    # predicting class name
    y_label = class_names[np.argmax(probability_cat_dog)]
    y_confidence = (100 * np.max(probability_cat_dog)).round(3)

    return y_label, y_confidence



def get_sample_images_link(dogname, num=6, df= pd.read_csv('static/dog_sample_images_path.csv')):
  """
  This function will return number of num images link from giving dog name.

  Parameters
  ----------
  dogname: string; a dog breed 
  num: int; number of images link
  df: dataframe; dog_sample_images_path.csv (default)
                        
  Return 
  ----------
  images_list: list; list contain 'num' of 'dogname' images link.
  """
  try:
    images_list = df[df['name']==dogname]['path'].sample(n=num).tolist()
  except:
    images_list = df[df['name']==dogname]['path'].sample(n=num, replace = True).tolist()
  return images_list



def top_three_images(most_likely_breeds_list):
    """
    This function will get dog image links from the Stanford dog dataset
    and capitalize the first letter in all dog names.
    
    Parameters
    ----------
    most_likely_breeds_list: list of string; list contains dog names
                            
    Return 
    ----------
    title_most_likely_breeds_list: list; a string where the first character in every word is upper case
    image_links_list: list of lists; each list contains the same dog breed image links
    """
    # remove the label number (ex:'n02108915-french_bulldog' to 'french_bulldog')
    # get the list of list dog images for each dog in the 'most_likely_breeds_list'
    image_links_list = [get_sample_images_link(i.split('-',1)[1]) for i in most_likely_breeds_list]
    # capitalize first letter in each words
    title_most_likely_breeds_list = [(x.split('-',1)[1]).replace('_',' ').title() for x in most_likely_breeds_list]

    return title_most_likely_breeds_list, image_links_list



def get_a_dog_image_from_dogtime(name):
    """
    This function will get a dog breed and return an associated image link from the DogTime.
    
    Parameters
    ----------
    name: string; a dog breed
                            
    Return 
    ----------
    dogtime_image_link: string; an image of the input dog breed from DogTime.
    """
    dogtime_df = pd.read_csv('static/dogtime2.csv')
    df = dogtime_df[dogtime_df['breed']==name]
    dogtime_image_link = df['image_src'].unique()[0]

    return dogtime_image_link


def get_dogtime_web_link(name):
    """
    This function will get a dog breed and return an associated URL link from the DogTime.

    Parameters
    ----------
    name: string; a dog breed
                            
    Return 
    ----------
    dogtime_web_link: string; DogTime Url for the input dog
    """
    dogtime_df = pd.read_csv('static/dogtime2.csv')
    df = dogtime_df[dogtime_df['breed']==name]
    dogtime_web_link = df['dog_page'].unique()[0]

    return dogtime_web_link





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

        # get number of face
        any_face = face_detect.faceDetector(filename, UPLOAD_FOLDER, DEST_FOLDER)

        uploaded_image_path = (os.path.join(UPLOAD_FOLDER, file.filename))

        # first model to predict cat or dog with confidence score
        catordog, catordog_confidence = cat_or_dog(uploaded_image_path)

        # top 3 most likely dog breed name and probability for each 3 dog breed
        most_likely_breeds_list, most_likely_probability_list = dog_breed_prediction(uploaded_image_path)

        # capitalize first letter in each words in 'most_likely_breeds_list' and get dog breed image links 
        most_likely_breeds_list, pic_path_list = top_three_images(most_likely_breeds_list=most_likely_breeds_list)

        dogtime_df = pd.read_csv('static/dogtime.csv')
        # get all dog breeds from Dogtime
        dog_breed_all = dogtime_df['breed'].unique()

        # check if the top predicted dog breed in DogTime
        if (most_likely_breeds_list[0] in dog_breed_all):
            # filer only with dog name = most_likely_breeds_list[0]
            df = dogtime_df[dogtime_df['breed']== most_likely_breeds_list[0]]
            characteristics_stars_info =list(zip(list(df['characteristic']), list(df['star'])))

            return render_template('view.html', filename=filename, catordog=catordog,catordog_confidence=catordog_confidence,
                                    which_breed=most_likely_breeds_list[0],in_df = True, any_face=any_face,
                                    pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list,
                                    dog_time_info=characteristics_stars_info)

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
        # get the drag uploaded image name after called function 'drag_save()'
        global DRAG_UPLOAD_NAME

        uploaded_image_path =os.path.join(app.config['UPLOADED_PATH'], DRAG_UPLOAD_NAME)
        any_face = face_detect.faceDetector(DRAG_UPLOAD_NAME, UPLOAD_FOLDER, DEST_FOLDER)

        # first model to predict cat or dog with confidence score
        catordog, catordog_confidence = cat_or_dog(uploaded_image_path)

        # top 3 most likely dog breed name and probability for each 3 dog breed
        most_likely_breeds_list, most_likely_probability_list = dog_breed_prediction(uploaded_image_path)

        # capitalize first letter in each words in 'most_likely_breeds_list' and get dog breed image links 
        most_likely_breeds_list, pic_path_list = top_three_images(most_likely_breeds_list=most_likely_breeds_list)
        
        dogtime_df = pd.read_csv('static/dogtime.csv')
        # get all dog breeds from Dogtime
        dog_breed_all = dogtime_df['breed'].unique()

        # check if the top predicted dog in DogTime
        if (most_likely_breeds_list[0] in dog_breed_all):
            # filer only with dog name = most_likely_breeds_list[0]
            df = dogtime_df[dogtime_df['breed']== most_likely_breeds_list[0]]
            characteristics_stars_info =list(zip(list(df['characteristic']), list(df['star'])))

            return render_template('view.html', filename=DRAG_UPLOAD_NAME, catordog=catordog,catordog_confidence=catordog_confidence,
                                    which_breed=most_likely_breeds_list[0],in_df = True, any_face=any_face,
                                    pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list,
                                    dog_time_info=characteristics_stars_info)

        return render_template('view.html', filename=DRAG_UPLOAD_NAME, catordog=catordog, catordog_confidence=catordog_confidence,
                                which_breed=most_likely_breeds_list, in_df = False, any_face=any_face,
                                pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list)




# the third way to upload an image from uploaded folder, so there are many repeated codes
@app.route('/gallery',methods=['GET','POST'])
def get_gallery():

    uploads_path = [x for x in os.listdir('static/uploads')if (x.split('.')[-1]).lower() in ALLOWED_EXTENSIONS ]
    index = (range(1,len(uploads_path)+1))
    uploads_path = list(zip(index, uploads_path))


    if request.method == 'GET':
        return render_template('display_uploads.html', uploads_path = uploads_path )
    else:
        # extract the value of submit button from request
        user_clicked_image_name =request.form.get('submitbutton')
        user_clicked_image_path = (os.path.join(UPLOAD_FOLDER, user_clicked_image_name))

        # get number of face
        any_face = face_detect.faceDetector(user_clicked_image_name, UPLOAD_FOLDER, DEST_FOLDER)
        catordog, catordog_confidence = cat_or_dog(user_clicked_image_path)
        most_likely_breeds_list, most_likely_probability_list = dog_breed_prediction(user_clicked_image_path)
        most_likely_breeds_list, pic_path_list = top_three_images(most_likely_breeds_list=most_likely_breeds_list)
        
        dogtime_df = pd.read_csv('static/dogtime.csv')
        dog_breed_all = dogtime_df['breed'].unique()

        # check if the top predicted dog in DogTime
        if (most_likely_breeds_list[0] in dog_breed_all):
            # filer only with dog name = most_likely_breeds_list[0]
            df = dogtime_df[dogtime_df['breed']== most_likely_breeds_list[0]]
            characteristics_stars_info =list(zip(list(df['characteristic']), list(df['star'])))

            return render_template('view.html', filename=user_clicked_image_name, catordog=catordog, catordog_confidence=catordog_confidence, which_breed=most_likely_breeds_list[0],
                                    in_df = True, any_face=any_face, pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list,
                                    dog_time_info=characteristics_stars_info)

        return render_template('view.html', filename=user_clicked_image_name, catordog=catordog, catordog_confidence=catordog_confidence, which_breed=most_likely_breeds_list,
                                in_df = False, any_face=any_face, pic_path_list=pic_path_list, most_likely_breeds_list=most_likely_breeds_list, most_likely_probability_list=most_likely_probability_list)
   

@app.route('/doginfo/', methods=['GET', 'POST'])
def doginfo():
    results = pd.read_csv('static/dogtime.csv')
    dog_breed_all = results['breed'].unique()
    dog_breed_all = np.sort(dog_breed_all)

    if request.method == 'GET':
        return render_template('doginformation.html', dog_breed_all=dog_breed_all)
    else:
        which_dog = request.form['dogs']
        graphJSON, dogpic = dogtime_barcharts.dogtime_plot(which_dog = which_dog)

        return render_template('doginformation.html', graphJSON=graphJSON, dog_breed_all=dog_breed_all, select=which_dog, dogurl = dogpic)


@app.route('/findyourdog', methods=['GET', 'POST'])
def findyourdog():
    if request.method == 'GET':
        return render_template('findyourdog.html', dogmap= dog_recommendation.prepare_recommendation_df()[1])
    else:
        # get user-selected characteristics, check 'dog_recommendation.py' in 'src' folder for detailed information;
        # 'selected_26_characteristics' list contains 26 elements;
        # each element has a user-selected characteristic number (0-5) 
        # index 0 means characteristic 'a', which means 'Adapts Well To Apartment Living.'
        selected_26_characteristics= np.asarray(request.form.getlist('characteristic_slider'))

        # check function 'dog_recommendation()' in 'dog_recommendation.py' for detailed information
        characteristics = dog_recommendation.make_characteristics_map(selected_26_characteristics)

        # get list of matching dog for recommendation
        recommendation_list, selected_26_characteristics = dog_recommendation.find_dog_recommendation(**characteristics)

        # 0: no matching recommended dog 
        find_any = bool(recommendation_list)

        # list of the recommended dog image link from DogTime
        dog_picture_links = [get_a_dog_image_from_dogtime(dog) for dog in recommendation_list]
        
        # dog web link
        dog_web_links = [get_dogtime_web_link(dog) for dog in recommendation_list]

        dog_recommendations = list(zip(recommendation_list, list(zip(dog_picture_links,dog_web_links))))

        return render_template('findyourdog.html', dogmap=dog_recommendation.prepare_recommendation_df()[1], recommendation_list=recommendation_list, find_any=find_any, dog_recommendations=dog_recommendations, selected_26_characteristics=selected_26_characteristics)


if __name__ == "__main__":
    app.run()



# export FLASK_ENV=development; flask run
