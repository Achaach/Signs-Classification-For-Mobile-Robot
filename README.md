# Signs-Classification-For-Mobile-Robot


In this project we create an algorithm to classify the signs for future mobile robot image recognition.
We design a solution to identify the signs using KNN and OpenCV following the steps below:

• Crop the image to focus classification only on the sign portion instead of the entire image.

• Incorporate color

• Incorporate other high-level features

After the classification, we will apply the algorithm to our mobile robot Turtlebot 3.


### Note

It will train and test each time you run the code

classification.py - This code runs considering the cropped image by the orange rectangular in each image and considers the 3 channels. The KNN classifier acts only on the raw picture pixel intensity values. The resulting model achieves around 0.95 accuracy (much better than random guessing, 1/4).

There are two ways to run this code:

1) With visualization of fed images:
python classification.py -v

2) Without visualization of fed images:
python classification.py
