# PIC16B_GroupProject - Dog Facial Recognition
<img src="https://humanepro.org/sites/default/files/styles/article_new/public/images/post/Scoop_FaceValue_cover.jpg" alt="dog facial recognition" width="350"/>

### Group Member:
- Peng Wu
- Weixin Ye
- Miao Wang
- Jiamu Liu


## Short Version of This Project

This project exceeds the Heroku slug size (500MB). Therefore, we created another short version of this project in the [testing project repository](https://github.com/PengWu2626/testing_project). The short version of this project moved face detecting and drop zone.

**Warning**: the short version still has slug size (437 MB) exceeds the soft limit (300 MB) which may affect boot time.
It's easy to get H12 errors ( H12 errors occur when an HTTP request takes longer than 30 seconds to complete).
Sometimes you can close and re-open it to solve the problem. 

**Webapp Link:**
https://pic16b-dog.herokuapp.com



## Instructions for M1 Mac to Run This Project Locally

For Apple Silicon Mac:
Our web app includes the TensorFlow package. To use TensorFlow on M1 Mac, we need to use "tensorflow-metal PluggableDevice."
First, watch this [Youtube video](https://www.youtube.com/watch?v=Qu1QitU6GXA)  to install and create a Miniforge Environment.
This video is based on the instruction from the [apple developer](https://developer.apple.com/metal/tensorflow-plugin/)

The video created the environment named `tfm1`
After all instructions, open your terminal, then run the command `conda activate tfm1`.
Inside that environment, 
we need to install these packages:

- `pip3 install --upgrade pip`

- `pip3 install flask`

- `pip3 install plotly`

- `pip3 install opencv-python`

- `pip3 install matplotilb`

- `pip3 install Flask-Dropzone`

- `pip3 install pandas`

- `pip3 install imutils`

You can use  Visual Studio Code to open the folder for this whole project.

After you open this project folder, find `app.py` file; there is a terminal inside the Visual Studio Code,

run `conda activate tfm1` ( we used the example name from the video),

then run `export FLASK_ENV=development; flask run`

