## Writeup Template

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
[image1]: ./examples/car_not_car.png
[image2]: ./examples/output_bboxes.png
[image3]: ./examples/labels_map.png
[image4]: ./examples/test1_example_bb.jpg
[image5]: ./examples/test1_example_heatmap.jpg
[image10]: ./examples/sliding_windows.jpg
[image11]: ./examples/HOG_example.jpg
[frame1]: ./examples/frame1.png
[frame2]: ./examples/frame2.png
[frame3]: ./examples/frame3.png
[frame4]: ./examples/frame4.png
[frame5]: ./examples/frame5.png
[frame6]: ./examples/frame6.png
[last_f]: ./examples/last_frame.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

To start this project, I merged the structure of the Lane Finding project. Similar to my implementation there, I started adding to a cli implemented with clize.

```
./cli.py --help
Usage: ./cli.py command [args...]

Commands:
  full-run
  test-calibrate-camera
  pickle-camera-calibration
  lane-line-single-image
  vehicle-detect-data-summary
  vehicle-detect-train
  vehicle-detect-single-image
  vehicle-detect-full-run
  vehicle-detect-hog-sample
```

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images, as seen in the `Classifier._get_dataset` method in `vehicle_detection/classifier.py`.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

HOG features are extracted in `vehicle_detection/features.py`. The function `hog_features_by_channel` appends the channels specified in the `HOG_CHANNEL` module constant.

I explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image11]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various parameters with the intention of maximizing the test accuracy from the `LinearSVC` classifier. Ultimately I had the best luck in test accuracy and fewest false positives (in this particular sample) when using HLS with accuracy of .9896, 11 orientations, and the original pixels per cell (8x8) and cells per block (2x2).

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Because the compilation of features for an image is shared by the classifier and the sliding window, related functions are organized into `vehicle_detection/features.py`. The `to_features` function takes an image and returns HOG, spatial, and color histogram features.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I tried multiple approaches, ranging from very little overlap to additional overlap. Too much vertical overlap caused lots of noise in the downstream heatmap. I ultimately settled on 3 fixed sizes of window with increasing overlaps that seemed to work well for me.

In my original implementation, each window size also walked the frame starting at the halfway point (`MIN_Y` in `vehicle_detection/detector.py`) of the image. In this final implementation, each size only slides at a fixed Y coordinate.

Here's an example of the sliding image search with the detected vehicles:

![alt text][image10]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using HLS 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. After the pipeline was roughly working from a vehicle identification standpoint I worked to remove false positives by tweaking the heatmap to be less permissive. As this evolved, the heatmap came to store all window history for the past N frames (40 in this version), and a threshold of 30 was required for the heatmap to identify a vehicle. 

Here are some example images:

![alt text][image4]
![alt text][image5]

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

This code is located in `vehicle_detection/detector.py` in the `Detector._get_estimated_positions` method.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames, their corresponding heatmaps, and the output of `scipy.ndimage.measurements.label()`:

![alt text][frame1]
![alt text][frame2]
![alt text][frame3]
![alt text][frame4]
![alt text][frame5]
![alt text][frame6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][last_f]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* Performance is a big issue! :( I worked on calling `skimage.hog()` once per frame, but it would have required modifications to the rest of my code that I needed to plan out more. Selecting cells from the results didn't match the way I had already built the sliding window.

* There's still a fair number of false positives. I read up on hard negative mining and given more time would explore that.

* I'd love to explore integrating the Udacity datasets (and other datasets!) to create more powerful detection.

* This was a really fun project!
