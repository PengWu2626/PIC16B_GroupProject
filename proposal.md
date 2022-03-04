# PIC16B_GroupProject - Dog Facial Recognition
<img src="https://humanepro.org/sites/default/files/styles/article_new/public/images/post/Scoop_FaceValue_cover.jpg" alt="dog facial recognition" width="350"/>

### Group Member:
- Peng Wu
- Weixin Ye
- Miao Wang
- Jiamu Liu

## Abstract
There are way too many dog breeds, and many dogs are similar. So this project will build an algorithm to identify the user-uploaded images to classify the dog's breeds. Moreover, it will prompt the users with some essential characteristics and features of the dog breeds to get to know the dog better. 
Our initial approach:

### Prepare data
1. Looking for appropriate datasets
2. Doing exploratory analysis
3. Cleaning and preparing related data
4. Incorporating the data into the database

### Pick and train the model
5. Checking iamges validity
6. Picking a proper model
7. Splitting training and testing data
8. Training and fitting the model
9. Visualizing the result and confusion matrix

### Test result
10. Testing and predicting the test data
11. Testing the images of my french bulldog for fun

## Planned Deliverables
**Full success**: We will build a web app that allows users to upload the dog images and show the possible breeds. This web app should warn the user if the image is not a dog.

**Partial success**:  In the first stage, we will accept images with one dog and identify only a few certain breeds. Instead of developing a web app, we will use Jupyter notes to show our core code and explain how to construct the machine-learning algorithm.

## Resources Required
We need datasets that contain a variety of dog breeds, and here are two of them which we might use.
- https://www.kaggle.com/kingburrito666/largest-dog-breed-data-set
- https://www.kaggle.com/c/dog-breed-identification

## Webapp framework
We will use [Flask](https://en.wikipedia.org/wiki/Flask_(web_framework)) to build our web application and deploy it on [Heroku](https://www.heroku.com/) platform.

## Tools and Skills Required
- [TensorFlow](https://www.tensorflow.org/) package and Convolutional Neural Networks (CNN) will be employed to classify and extract features from images. We might use more than one model to compare the accuracy. One of our models might be from the sequential model, `tf.keras.models.Sequential`. 

- Database(e.x. SQLite) and related techniques will be employed to avoid OOM(out of memory).


## What You Will Learn
- Deep learning in classifying image-related problems - In the future, we could also build similar projects such as gender or other species identification programs.

- Neural Networks Technique - Learn how to use Convolutional Neural Networks (CNN) algorithm to assign important labels to images and differentiate one from another. 

- TensorFlow - Learn the similarity between TensorFlow and Numpy array and how layers act as a function that takes in one TensorFlow and spits out another TensorFlow with different shapes. 

- Database techniques - Learn how to create a database; import, filter, sort, and export data; dataset visualization. For example, querying a record from a database could be an essential skill we will learn.

- Version control - Learn how to develop a comprehensive project with the team concurrently via Git and Github. For example, git clone, add, stage, commit, fetch, pull, rebase, creating branch, and solving merging conflicts could be essential skills we will learn.

- App Development - Learn how to develop a simple app to fulfill end-user needs.


## Risks
The first risk is that we have over one hundred species, so we are not sure about the accuracy of detecting similar species. Moreover, we might include too many features that cause the overfitting since many species. Also, there are a lot of mix-breed dogs, which will make our project harder.

The second risk is that extracting the images that contain multiple dogs or fake dogs might require a more complex algorithm.

The third risk is maintaining devices compatibility. Our app may not run on specific devices or platforms since there are too many operating system versions.


## Ethics
1. Any people, especially dog lovers, can use this product to help them identify their images of the dog. People can take a picture of the dog they like on the street and check the breed immediately. 
2. Since machine learning sometimes leads to error, it tends to "translate" some rare breeds to popular breeds. Thus, the famous breeds have more presence in front of humans, resulting in a mere exposure effect. Dogs that belong to rare breeds would gradually leave people's sight. We hope to set a threshold and only show the result if the app is confident with the picture's content.
3. 
    - Different dog breeds may have various behaviors. Understanding their behaviors may help owners know and train their pets better.
    - A dog breed will determine what their drive will be. Without proper outlets, dogs can become a nuisance. Therefore, knowing this may help you figure out if an owner's house will fit them.
    - To identify the breed of the dog is important. Some people they do not know about dogs, so they buy or adopt a dog just because the the pictures of them are cute or pretty. Later they may abandon them because of their behavior. If they can know what breed the dog is and what characters they have, it will help them make right decisions.


## Tentative Timeline
- 1~2 weeks:
  - Find the appropriate dataset
  - Find the breeds(may be 5) that we want to identify(start with a easy task)
  - Exploratory analysis 
  - Cleaning and preparing the dataset
  - Incorporating the data into the database
- 3~4 weeks:
  - Checking the validity of images 
  - Pick the model
  - Split Training and testing data
  - Training and fitting the model
- 5~6 weeks:
  - Visualizing the result and confusion matrix
  - Testing and predicting the test data
  - Testing the images of my french bulldog for fun
  - Thinking about how to improve our model
  - Increase our breeds idfentification number