# PIC16B_GroupProject - Dog Facial Recognition

PIC 16B Group Project

Group Member:
- Peng Wu
- Weixin Ye
- Miao Wang
- Jiamu Liu


## Abstract
There is way too many dog breed, and many dogs are similar. So this project will build an algorithm to identify the user-uploaded images to classify what the dog is. Moreover, it will help them make right decisions before they want them.
Our initial approach:
1. Find the appropriate dataset
2. Exploratory analysis
3. Cleaning and preparing the dataset.
4. Incorporating the data into the database
5. Checking the validity of images 
6. Pick the model
7. Split Training and testing data
8. Training and fitting the model
9. Visualizing the result and confusion matrix
10. Testing and predicting the test data
11. Testing the images of my french bulldog for fun

## Planned Deliverables
**Full success**: We will build a local app that allows users to upload the images and show the possibility breed of the dogs. This local app should also prompt the user if the image is not the dog.

**Partial success**:  In the first stage, we will identify a few breeds and suppose users will upload images with no more than one dog. We will not build a local app. Instead, we will use Jupyter notes to show our code and explain how we construct the machine-learning algorithm to classify the dog breed. 

## Resources Required
We need a dataset that contains a variety of dog breeds. We will use the data set from Kaggle. 
- https://www.kaggle.com/kingburrito666/largest-dog-breed-data-set
- https://www.kaggle.com/c/dog-breed-identification

Those datasets include most dog species which reduced the possibility of verifying unknown breeds from the user. 

## Tools and Skills Required
We will use the TensorFlow package and Convolutional Neural Networks (CNN) to classify and extract features from images. We might use more than one model to compare the accuracy. For example, one of our models might be from the sequential model, `tf.keras.models.Sequential`
Since our dataset has many images, we need to use the database to help us manage and avoid the memory problem on our computer; we also need to know some SQL techniques to access the data.



## What You Will Learn
After completing this project, we will learn how to apply the techniques from deep learning in classifying image-related problems. We also can build similar projects such as gender or other species identification. This project will show how to use Convolutional Neural Networks (CNN) algorithm to assign importance labels to images and differentiate one from the other. You will know how tensor is pretty similar to a Numpy array and how layers act as a function that takes in one tensor and spits out another tensor with different shapes. Finally, you will learn how to create a complex dataset visualization too. 


## Risks
The first risk is that we have over one hundred species. Therefore, we are not sure of the accuracy of detecting similar species; moreover, we might include too many features that cause the overfitting since there are many species. Also there are a lot of mix breed dogs, it will make our project harder.

The second risk is that extracting the images that contain multiple dogs or fake dogs might require a more complex algorithm.

## Ethics
1. Any people, especially dog lovers, can use this product to help them identify their images of the dog. People can take a picture of the dog they like on the street and check the breed immediately. 
2. We do not think any group of people will be harmed from using this product.
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


