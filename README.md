**Group Contributions Statement**

**Peng Wu**:  
- [DogTime web scrapper](https://github.com/PengWu2626/PIC16B_GroupProject/tree/main/DogTime_scraper)
- [get_sample_images_path.ipynb](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/data/get_sample_images_path.ipynb): This python code generated a CSV file that contains URL links for all images in the Stanford Dogs Dataset.
- [dog_recommendation.py](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/src/dog_recommendation.py): get all user-selected number of 26 characteristics, and find out all matched dog breeds.
- [dogtime_barcharts.py](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/src/dogtime_barcharts.py): make bar charts of input dog breed
- part of the [base.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/base.html)
- most part of the [view.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/view.html): prediction `view` page
- most part of the [app.py](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/app.py) file.
- [findyourdog.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/findyourdog.html): `Find Your Best Dog` page (an extra feature that is not in our initial proposal)
- [drag_upload.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/drag_upload.html): `Drag Upload` page (an extra feature that is not in our initial proposal)
- [doginformation.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/doginformation.html): `DogTime Information` page (an extra feature that is not in our initial proposal)
- [display_uploads.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/display_uploads.html): `Click Gallery Upload` page (an extra feature that is not in our initial proposal)



**Weixin Ye**:
- created and analyzed the [dog breed classification model](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/data/Model1.ipynb) using Xception with sequential API.
- created and analyzed [another dog breed classification model](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/data/Model2.ipynb) using Xception with functional API.
- part of the [intro.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/intro.html): `Introduction` page
- part of the [index.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/index.html)



**Miao Wang**:
- draft of [suggestion.py](https://github.com/PengWu2626/pic16b_group_project_short_version/blob/main/src/suggestion.py) to find the matched dogs.
- dog breed classification model 
- another dog breed classification model 


**Jiamu Liu**:
- [face_detect.py](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/src/face_detect.py): `face detection model`
- part of the [view.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/view.html)
- part of the [index.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/index.html)
- part of the [base.html](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/templates/base.html)
- part of the [app.py](https://github.com/PengWu2626/PIC16B_GroupProject/blob/main/app.py)




# PIC16B_GroupProject - Dog Facial Recognition

<img src="https://humanepro.org/sites/default/files/styles/article_new/public/images/post/Scoop_FaceValue_cover.jpg" alt="dog facial recognition" width="350"/>

### Group Member:

- Peng Wu
- Weixin Ye
- Miao Wang
- Jiamu Liu

## Full Version of This Project

The full version has `slug size (482 MB) exceeds the soft limit (300 MB)` which may `affect boot time`.
It's easy to get H12 errors ( H12 errors occur when an HTTP request takes longer than 30 seconds to complete).
Sometimes you can close and re-open your browser to solve the problem.

This Web app uses a `lot of memory`, and it might be `crashed` if you use pictures to predict too many times at once. I suggest trying two images of predicting, then closing your browser and re-open it.

**Full Version Webapp Link**:

https://pic16b-dog-facial-recognition.herokuapp.com/

## Short Version of This Project

This project exceeds the Heroku slug size (500MB). Therefore, we created another short version of this project in the [pic16b_group_project_short_version](https://github.com/PengWu2626/pic16b_group_project_short_version) repository. The short version of this project removed face detecting and drop zone.

```diff
- Warning:
```

the short version still has `slug size (435 MB) exceeds the soft limit (300 MB)` which may `affect boot time`.
It's easy to get H12 errors ( H12 errors occur when an HTTP request takes longer than 30 seconds to complete).
Sometimes you can close and re-open your browser to solve the problem.

This Web app uses a `lot of memory`, and it might be `crashed` if you use pictures to predict too many times at once. I suggest trying two images of predicting, then closing your browser and re-open it.

Some errors you might encounter while using this web app:

```diff
- Error H10 (App crashed. A crashed web dyno or a boot timeout on the web dyno will present this error.)
- Error H12 (Request timeout)
- Error R14 (Memory quota exceeded)
- Error R15 (Memory quota vastly exceeded)

```

**Short Version:**

https://pic16b-dogfr-short.herokuapp.com

## Instructions for M1 Mac to Run This Project Locally

For Apple Silicon Mac:
Our web app includes the TensorFlow package. To use TensorFlow on M1 Mac, we need to use "tensorflow-metal PluggableDevice."
First, watch this [Youtube video](https://www.youtube.com/watch?v=Qu1QitU6GXA) to install and create a Miniforge Environment.
This video is based on the instruction from the [apple developer](https://developer.apple.com/metal/tensorflow-plugin/)

The video created the environment named `tfm1`
After all instructions, open your terminal, then run the command `conda activate tfm1`.
Inside that environment,
we need to install these packages:

- `pip3 install --upgrade pip`

- `pip3 install flask`

- `pip3 install plotly`

- `pip3 install opencv-python`

- `pip3 install matplotlib`

- `pip3 install Flask-Dropzone`

- `pip3 install pandas`

- `pip3 install imutils`

You can use Visual Studio Code to open the folder for this whole project.

After you open this project folder, find `app.py` file; there is a terminal inside the Visual Studio Code,

run `conda activate tfm1` ( we used the example name from the video),

then run `export FLASK_ENV=development; flask run`
