# Vehicle-Detection
The goal of this project is to use computer vision techniques and Histogram of Oriented Gradients to detect vehicles in images and track their position across frames in a video stream.

![Final Result Gif](https://github.com/JustinHeaton/Vehicle-Detection/blob/master/images/result1.gif)

The project is completed in the following stages:
* **Step 1**: Create a function to draw bounding rectangles based on detection of vehicles.
* **Step 2**: Create a function to compute Histogram of Oriented Gradients on image dataset.
* **Step 3**: Extract HOG features from training images and build car and non-car datasets to train a classifier.
* **Step 4**: Train a classifier to identify images of vehicles.
* **Step 5**: Identify vehicles within images of highway driving.
* **Step 6**: Track images across frames in a video stream.

### Code:
This project requires python 3.5 and the following dependencies:
- [NumPy] (http://www.numpy.org/)
- [Pandas] (http://pandas.pydata.org/)
- [Scikit-learn] (http://scikit-learn.org/stable/)
- [Scikit-image] (http://scikit-image.org/docs/dev/api/skimage.html)
- [matplotlib] (http://matplotlib.org/)
- [OpenCV] (http://opencv.org/)
- [MoviePy] (http://zulko.github.io/moviepy/)

### Step 1: Drawing Bounding Rectangles

For this step I defined a function `draw_boxes` which takes as input a list of bounding rectangle coordinates and uses the OpenCV function `cv2.rectangle()` to draw the bounding rectangles on an image.

![Bounding Boxes](https://github.com/JustinHeaton/Vehicle-Detection/blob/master/images/boxes.jpg)

### Step 2: Compute Histogram of Oriented Gradients

For this step I defined a function `get_hog_features` which uses the OpenCV function `hogDescriptor()` to get Histogram of Oriented features from an image. The functions computes the HOG features from each of the 3 color channels in an image and returns them as a single feature vector. 

![HOG images](https://github.com/JustinHeaton/Vehicle-Detection/blob/master/images/hog1.jpg)

### Step 3: Extract HOG features and build training datasets

For this step I defined a function `extract_features` which calls the function `get_hog_features` and appends the HOG features from each image to a dataset of training features. After building the datsets of car and noncar images and labelling them appropriately I normalized the features to have zero mean and unit variance using the Scikit-learn function `StandardScaler`, and used `train_test_split`to split them in to training and testing datasets. 

After testing the performance of the HOG features from several color spaces including:
* Lab
* HSV
* YUV
* LUV

I found the YUV color space to have the best performance for identifying vehicles and I chose to use that space for feature extraction.

### Step4: Training a classifier

The next step is to train a classifier which will be used to identify vehicles within images. I chose to evaluate the performance of the following three classifiers:
* Linear Support Vector Machine
* Logistic Regression Classifier
* Multi-layer Perceptron

These were the results of training and testing on each one:

|Classifier|Training Accuracy|Test Accuracy|Prediction Time|
|----------|-----------------|-------------|---------------|
|LinearSVC |1.00|.983|.0015 seconds|
|Logistic Regression|1.00|.984|.0002|
|Multi-layer Perceptron|1.00|.993|.0007|

Based on the above results I decided to move forward with the Multi-layer Perceptron as my classifier to identify vehicles in images and in the video stream. Althought the Logistic Regression Classifier was a bit faster in making predictions, I chose to favor the MLP classifier due to the increase in prediction accuracy.

### Step 5: Identifying vehicles in images

To search for vehicles within images I chose to implement a sliding window approach where I looked at one slice of the image at a time and made predictions on the HOG features from that particular window. In order to minimize the search area and speed up the pipeline I only searched for cars in the lower half of the image. Additionally, my algorithm searches for vehicles in windows of multiple scales, with an 80% overlap, in order to identify vehicles which can be either near or far in the image and will appear to have different sizes.

In order to ensure a high confidence for my predictions and minimize the instance of false positives, I made use of the MLP classifier's method `predict_proba` which returns a probability score for each possible class. I chose to threshold my predictions by looking for windows which were classified as vehicle with a probability score higher than 0.99.

The coordinates of windows which are classified as vehicle will be appended to a list called `detected` and after all windows are searched, I will use the `draw_boxes` function to draw the boxes in `detected` on to a blank `mask` image with the same dimensions as the input image. 

Next I use the OpenCV function `cv2.findContours` to find all of the objects in the `mask` image and once the contours are found I used the OpenCV function `cv2.boundingRect` on each contour to get the coordinates of the bounding rect for each vehicle.

Finally, I create a copy of the original image, called `result`, on which I draw the bounding rectangles. Below you can see the process on an example image:

![Annotated Cars](https://github.com/JustinHeaton/Vehicle-Detection/blob/master/images/cars.jpg)

### Step 6: Tracking images across frames in a video

Finally, to track vehicles across frames in a video stream, I decided to create a class `boxes` to store all of the bounding rectangles from each of the previous 12 frames in a list. In each frame, I then combine the lists of bounding rectangles from current and previous frames, and then use the OpenCV function `cv2.groupRectangles` to combine overlapping rectangles in to consolidated bounding boxes. Within the group rectangles function I set the parameter `groupThreshold` equal to 10 which means that it will only look for places where there are greater than 10 overlapping boxes and it will ignore everything else. 

The group rectangles function takes care of the problem of false positives because if any part of the image is classified as a vehicle in fewer than 10 out of 12 consecutive frames, it will be filtered out and will not be included in the final bounding rectangles which are annotated on to the output video. 

![Final Result Gif](https://github.com/JustinHeaton/Vehicle-Detection/blob/master/images/result1.gif)

### Discussion: 

The most difficult part of this project, and possibly the most important, is the elimination of false positives. Any time a non-vehicle object is falsely identified as a vehicle it can influence the car to make driving decisions which are potentially hazardous such as swerving or slamming on the brakes. 

My pipeline, as it is now, is still falsely identifying some non-vehicle objects such as trees and traffic signs. In order to improve performance further it will be necessary to bolster the training dataset by adding more negative features based on the false positives which were identified in the video stream. 

Switching to the OpenCV HOG function from the skimage version greatly improved performance, but it is still not quite up to real time. Right now my pipelie procseses videos at about 4 frames per second, and to be useful in real time it would need to be closer to 25 or 30. Moving forward I will attempt to prune the number of features and reduce the number of windows searched in an attempt to speed up the prediction process. Additionally, deep learning methods might prove to be more useful and higher performing than computer vision and I plan to explore this as well.
