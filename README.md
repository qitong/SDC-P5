# **Vehicle Detection Project, Author: Qitong Hu**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

### Submission Content
* CARND-VEHICLE-DETECTION.ipynb, the notebook is main entrance of this project
* lesson_functions.py, it defines functions mainly from Udacity courses, I made small modifications on some of them.
* helper_functions.py, functions to display images and I/O related ones.
* training/ folder contains training images of cars and notcars, the full set from Udacity Vehicle Tracking S3 bucket
* test_images/ folder contains those images for test.


## Histogram of Oriented Gradients (HOG)

### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.hER
I started by reading in all the `vehicle` and `non-vehicle` images. Those images are stored in training/ foloder, which contains 2 subfolders named `cars` and `notcars`, which contains the full set of image in Udacity Vehicle Tracking S3 bucket respectively.  
Then to extract the HOG feature, I mainly used the function `get_hog_features()` studied in the couse which warpped `skimage.feature.hog` for its own. And the function to call that to show result is defined in notebook code cell 4 `show_hog_feature()`.  
Here are the examples of the HOG feature of Vehicles and Non-Vehicles:
![alt hog_vehicle](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/hog_vehicle.png)
![alt hog_non_vehicle](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/hog_non_vehicle.png)

### 2. Explain how you settled on your final choice of HOG parameters.
The main parameters for HOG are orient=9, pix_per_cell=8, cell_per_block=2, those parameters are defined in code cell 6.
I tried several combination during the course exercise, I think most orient parameters from 8 to 12 gives similar final result in training with SVM, I picked 9. I have tried with other numbers for other 2 paramters, while 8 and 2 respectively gives reasonable result, and since the image itself is 64x64, I think after applying 8 and 2 should make the hog matrix fit more to memory (I guess...)  
Other parameters which is related to the HOG feature functions (not that close) are color space which I use 'YUV' and channel to apply which I use 'ALL' of those 3, so that in the image above you'll see HOG of Y、U、V channel respectively.   


### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).
I use color features as well, since I can tell the difference from the histograms of cars and non-cars in most of cases, and for bin_spatial I think it is more original and supplement with details losed by color histogram and HOG. At least, with enough training set it won't do anything harmful for my classifier.  
To prepare the data set for training, I load in the image data, convert it to 'YUV' color space, applied the feature extraction in function `extract_features()` which called `single_img_features()` for individual processing (in `lesson_functions.py`). It extracts HOG feature of all 3 channel, bin spatial feature and color histogram feature (those parameters also defined in code cell 6 in notebook). After extracting the features, I applied standard_scaler from `sklearn.preprocessing` to normalize the data set-wise. Then, I added their y_target decided by which folder it comes from and shuffle them. Finally, split the whole training set to traing and validation at ratio 8:2. (Those code are in code cell 8).  
Then, I used these data to train a SVM classifer using 'RBF' kernel and default parameters. The classifer give me an accuracy of 99.6% which I am satisified with.  

![alt classifier_accuracy](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/classifier_accuracy.png)

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

