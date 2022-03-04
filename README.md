# PIC16B_GroupProject - Dog Facial Recognition
<img src="https://humanepro.org/sites/default/files/styles/article_new/public/images/post/Scoop_FaceValue_cover.jpg" alt="dog facial recognition" width="350"/>

### Group Member:
- Peng Wu
- Weixin Ye
- Miao Wang
- Jiamu Liu


## Instructions for M1 Mac to Run This Project Locally

For Apple Silicon Mac:
Our web app includes the TensorFlow package. To use TensorFlow on M1 Mac, we need to use "tensorflow-metal PluggableDevice."
First, watch this [Youtube video](https://www.youtube.com/watch?v=Qu1QitU6GXA)  to install and create a Miniforge Environment.
This video is based on the instruction from the [apple developer](https://developer.apple.com/metal/tensorflow-plugin/)

The video created the environment named `tfm1`
After all instructions, then `conda activate tfm1`.
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

type `conda activate tfm1` ( we used the example name from the video),

then type `export FLASK_ENV=development; flask run`

