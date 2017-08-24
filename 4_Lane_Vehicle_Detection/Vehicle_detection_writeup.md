##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/img1.png
[image2]: ./output_images/img2.jpg
[image2_1]: ./output_images/ori_8_pix_4_cell_2.png
[image2_2]: ./output_images/ori_12_pix_8_cell_2.png
[image2_3]: ./output_images/ori_10_pix_8_cell_6.png
[image2_4]: ./output_images/ori_2_pix_8_cell_9.png
[image5]: ./output_images/img5.png
[image6]: ./output_images/img6.png
[image7]: ./output_images/img7.png
[image8]: ./output_images/img8.png
[image8_1]: ./output_images/img8_1.png
[image8_2]: ./output_images/img8_2.png
[video1]: ./output_images/white.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  
## Many parts of my code is from Udacity Vehicle Detection module
---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The idea behind HOG feature is to identify shape of gradients by blocks in a given channel of image. Through HOG extraction, we are interested in distinguishing a car image from a non-car by looking at its edges.

The code is part of first section of submission.ipynb. 'get_hog_features' takes input of image, number of orientations, and size of cell and block. HOG will aggregate cell values and determine the gradient for each block.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `get_hog_features()` parameters.  

Here are examples of using HOG parameters of orientation, pixels, and cells
i)   9,  (8,8), (2,2)
![alt text][image2]
ii)  8,  (4,4), (2,2)
![alt text][image2_1]
iii) 12, (8,8), (2,2)
![alt text][image2_2]
iv)  10, (8,8), (6,6)
![alt text][image2_3]

####2. Explain how you settled on your final choice of HOG parameters.

After trying variations in parameters, I have found that lowering pix to 4 didn't help with extracting shape of car. For orientation and cell per block, I didn't notice much differences. It's because I didn't try extreme values and for our purpose I wanted to see if there were much differences between reasonable values. For instance, orientation of 2 would result in something bad like this:
![alt text][image2_4]

To really compare which parameter values are better, I suspect that training SVM over HOG features and comparing training accuracy might be better than to compare over HOG images. But, with (i) option of 9,(8,8),(2,2) we can identify car shape decently and fulfill our need.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

After extracting features from all data, I used SVM to train the features. Before training the data, the data was normalized using StandardScaler() from sklearn.preprocessing. Then these normalized data were splitted into train and test sets. (80% Training, 20% Test set). I trained a linear SVM using LinearSVC from sklearn and got test Accuracy of SVC =  0.98. Codes can be found at section 2, 'Train SVM' from submission.ipynb.

![alt text][image5]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I have decided to use a function, called "find_cars", that extracts HOG features from image and subsample for smaller them, instead of having to extract HOG feature for each window. Then, I drew rectangles on areas that are identified to be a car based on trained SVM. 

![alt text][image6]


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector. Here are some example images:

![alt text][image8]
![alt text][image8_1]
![alt text][image8_2]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/white.mp4)

![alt text][video1]

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections of latest three frames I created a heatmap and then set threshold of 3. For areas detected more than 3 times within the three frame would be shown in heatmap. Then, I used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I constructed bounding boxes to cover the area of each blob detected.  

![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

From observing output video, I have noticed that when two cars are ovelapped from camera, it stop recognizing the car behind the first. Forecasting the position of car would be necessary to avoid such cases. 

Also, when two cars are very close, the heap map would recognize the two boxes as one and project out a single box covering both cars instead of two. Beside just taking positions of box before heatmap process, I think adding properties to box could be a way to solve. For instance, in addition to flagging for car object, flagging for color of car could help when applying heatmap. But then, if black cars are side by side, flagging by color solely would be ineffective. 
