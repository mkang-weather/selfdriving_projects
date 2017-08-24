##Advanced Lane Lines

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image]: ./output_images/download.png
[image0]: ./output_images/download0.png
[image1_1]: ./output_images/download1_1.png
[image1]: ./output_images/download1.png
[image2]: ./output_images/download2.png
[image3]: ./output_images/download3.png
[image4]: ./output_images/download4.png
[image5]: ./output_images/download5.png
[image6]: ./output_images/download6.png

[video1]: ./project_video_laned.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README
Line numbers of code is based on Advanced-Lane-Finding.py

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I have used calibration images of chessboards in order to identify its corners and compared with their objective points to calculate calibration. I used cv2 calibratecamera function to derive mitx and dist from the img and corner points. Then based on mitx and dist from calibration, I can undistort images. 

![alt text][image]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
I have compared original and distortion-corrected image.
![alt text][image0]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I have applied 1) absolute gradient threshold (line 143) or 2) magnitude and directional gradient threshold to grayscaled image (line 160-186). While with 1) abs gradient threshold, it filters for gradient with respect to x and y axis, 2) magntiude threshold measures the size of gradient and directional threshold complements by adding directional filter. By setting 'or' for combinations of gradient threshold, we can filter for gradient of interest with certain abs of gradients in x and y direction or gradient with size and direction of interest. 

Then I have added a color filter based on hls of image. After converting to hls space, to remove variations by shadow and light conditions, I have selected s of the image and filtered. (line 259-266)

Lastly, in order to avoid noise from part of image not associated with lane, I have filtered for lane by applying filters function. (line 193)

![alt text][image1_1]

These are examples of binary image that went through threholds.

![alt text][image1]
![alt text][image2]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

Perspective Transformation was done through a function called 'perspectivetransform' (line 310). Based on points of source that I specified and destination to be displaced, it transforms the source image using cv2 getPerspectiveTransform, warpPerspective functions. 

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image3]
![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

From transformed images, I have firstly took half bottom of image and set base x-pixel location for left and right lane based on its histogram. Then, I have broken the images into a number of windows and took mean of nonzero pixels for each of the windows. Then based on set of mean values, I have fit polynomial function for left and right side of lane. (line 435)
![alt text][image5]


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Based on inverse of polynomial functions I derived, I can get x values for range of y. Then, I would covert these pixels into meters by multiplying x and y with 3.7/700 and 30/720 meters per pixel. Afterwards, I refit to get coefficients of polynomial now in terms of meters and followed a formula provided by Udacity to calculate radius of curvature. Further explanations on mathematical background behind this formula can be found at [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). (line 386)

In order to identify position of vehicle, I have used min and max of x-pixel from unwarped image and took average of it and compared with the center of image. After taking the difference, I have multiplied with meter per x-pixel to derive its position. (line 598).

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.
![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  
Here's a [link to my video result](./project_video_laned.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There were a lot of test and trials to make reasonable outputs for parameter values. Although steps and code provided by udacity helped me a lot of getting a big picture of how to develop overall process, I had tried 1) different values of thresholds, 2) a number of source values for prospective transformation, and 3) sanity checks.  

I have set a sanity check on curvature where following curvature to be similar to current curvature, but I found that it may not be effective for curvy roads in which curvature from one frame to next can change drastically. In [harder challenged video](./harder_challenge_video_laned.mp4) I've noticed that at steep curves it won't capture lanes correctly as it would ignore such kind of drastic change and continue to project the same curvature from previous frame. 

Also, in some cases, I found it to be sensitive to shade and lightness despite of using s from hls for grayscale. I think it would be interesting to study more what else can supplement s and perhaps making a combination for grayscale can improve perforamnces. In addition, I expect that there are better values for parameters than what I used. For example, from testing source points on test images, I've found that source points can vary by images and chosen what it looked good to be. Perhaps accumulating and deriving best source points could be helpful.
