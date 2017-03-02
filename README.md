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
* PS: if you want check the code, please unzip the images in the training/ folder first, since it surpass the 1000 files limit for submission, I zip it.
* PS2: for last submission, there is one rubric point "A method, such as requiring that a detection be found at or near the same position in several subsequent frames, (could be a heat map showing the location of repeat detections) is implemented as a means of rejecting false positives, and this demonstrably reduces the number of false positives. " that I missed. I add this part of code in "Time Sequence False Positive Rejection" section and since the nonlinear SVC takes too long I change it to LinearSVC.

## Histogram of Oriented Gradients (HOG)
---
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
I use color features as well, since I can tell the difference from the histograms of cars and non-cars in most of cases, and for bin_spatial I think it is more original and supplement with details losed by color histogram and HOG. At least, with enough training set it won't do anything harmful for my classifier. Those functions are defined in the lesson_functions.py, `bin_spatial()` and `color_hist()` respectively.  
To prepare the data set for training, I load in the image data, convert it to 'YUV' color space, applied the feature extraction in function `extract_features()` which called `single_img_features()` for individual processing (in `lesson_functions.py`). It extracts HOG feature of all 3 channel, bin spatial feature and color histogram feature (those parameters also defined in code cell 6 in notebook). After extracting the features, I applied standard_scaler from `sklearn.preprocessing` to normalize the data set-wise. Then, I added their y_target decided by which folder it comes from and shuffle them. Finally, split the whole training set to traing and validation at ratio 8:2. (Those code are in code cell 8).  
Then, I used these data to train a SVM classifer using 'RBF' kernel and default parameters. The classifer give me an accuracy of 99.6% which I am satisified with.  

![alt classifier_accuracy](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/classifier_accuracy.png)

## Sliding Window Search
---
### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?  

The sliding window code are set in `slide_window()` and part of `find_car_in_field()` in lesson_functions.py and `slide_window_with_variant_size()` and `find_car_in_one_shot()` in notebook code cell 14 and 18.  
The main idea is to use the SVM classifier to decide if those windows are car. And from the perspective effect, the car presents in low y position aka far from the point of view should be smaller and those near the bottom, high y, should be larger. Thus, the window scale should change accordingly.  
Actually, `slide_window()` and `slide_window_with_variant_size()` is a pair, `find_car_in_field()` and `find_car_in_one_shot()` is another pair. First pair is my first implementation which generate windows first and call HOG for each window, while second pair generate the HOG for the whole image and applies window on it. The second implementation is more efficient since it only compute HOG once.  
For `slide_window()` and `slide_window_with_variant_size()` implementation, the window sizes are 64, 100, 120 respectively, and the y start postion are 300, 350, 400, since the image with lower y position, the top of image, should be sky or other unrelated stuff. The overlap is set to 0.5 times of the length, move 32, 50, 60 in one step. (those are defined in code cell 14). The overlap is larger than the second implementation below because this implementation takes more time to finish thus I cannot use small overlap that generate more windows to compute.  
For `find_car_in_field()` and `find_car_in_one_shot()` implementation, the window scales are 1, 1.5, 2 respectively and the cell_per_step is set to 2 and thus, for each step, the window will slide for 2 cells.  
Here are the examples of slide windows generated: (first one is with 0.5 overlap and the later is 0.8 overlap):  
![alt slide_window_overlap_0.5](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/slide_window_overlap_0.5.png)
![alt slide_window_overlap_0.8](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/slide_window_overlap_0.8.png)


Those are the images show the difference between those 2 implementations:
![alt detect_window_method_1](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/detect_window_method_1.png)
![alt detect_window_method_2](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/detect_window_method_2.png)

### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 3 scales sliding windows (search from x = 400, y = 300) using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, then applied heat map and label to detect the car bound rectangle, which provided a nice result.

Here are some example images:
![alt pipeline_on_image_1](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/pipeline_on_image_1.png)
![alt pipeline_on_image_2](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/pipeline_on_image_2.png)

## Video Implementation
---

### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/project_out.mp4) in repo.
And here's the same one on youtube:
[![alt Project_output](https://img.youtube.com/vi/MULAa3_NNnM/0.jpg)](https://youtu.be/MULAa3_NNnM)


### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then applied a threshold of 2 to that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. (Those code are in lesson_functions.py, `heatmap_with_threshold()` and `draw_labeled_bboxes()`) I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from my test image, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last one of them:

#### Here are 8 frames and their corresponding heatmaps:

![alt box_detection_1](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/box_detection1.png)
![alt box_detection_2](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/heatmap1.png)
![alt heatmap_1](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/box_detection2.png)
![alt heatmap_2](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/heatmap2.png)

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all 8 frames:  
![alt label_1](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/label1.png)
![alt label_2](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/label2.png)


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt final_result](https://raw.githubusercontent.com/qitong/SDC-P5/master/example_outputs/final_frame_result.png)

---
## Time Sequence False Positive Rejection:  
Here is what I miss for last submission. I implement an algorithm to reject false positive detection that not presented constantly in continuous frame. Those code are set under `Time Sequence False Positive Rejection` section in the notebook.  
I keep track of video frames in a queue object which has a fixed size of 10. For every new frame, after it calculates the heatmap of that frame, it will compare it with the 10 heatmaps before, if at least `frame_thresh+1` frames contain the same point, it will then count as a true positive heat point and pipeline it to label functions, otherwise it will reject it. Note that the heatmap of current frame will then do a binarization (not sure the heat frequence but only count the occurence) and store into the queue pop the earliest one.  

---
## Discussion
---
### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
While, the most important problem is not where the pipeline fail but rather the time it takes to detect the car. It takes couple of hours to generate the whole video of less than 1 minute which cannot be directly use as a real time solution in the real world. Yet, it may be due to the computational power of my computer and could be improved by using a more powerful computer. Probably, if not changing the main body of the algorithm we could use some parallel programming, to compute the batch of windows simutaneously. Also we don't need to compute every frame, maybe sample every 1/2 second and during the period using "match_template()" or other low cost algorithms, since it should not change much (differ by speed) in between. 
Secondly, I think the classifier is highly depending on the training set, since I tried with smallest set and pipeline result is worse. Definitely, more data generate better result. And the feature extraction part relies on the somewhat "ad-hoc" designed (sometimes the car is too far to detect). I think deep learning with CNN architecture could be more useful for this situation. 

