## Writeup

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

[image1]: ./output_images/calibration1_undistort.jpg "Undistorted Calibration Image"
[image1_1]: ./output_images/undistort_test1.jpg "Undistorted Test Image"
[image1_2]: ./output_images/binary_test1.jpg "Binary Threshold Image"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./output_images/warped_threshold_undistort_straight_lines2_withlines.jpg "Warp Example"
[image5]: ./output_images/slide_windows.jpg "Fit Visual"
[image6]: ./output_images/warped_back_lines.jpg "Output"
[video1]: ./project_video_output.mp4 "Video Output"
[video2]: ./challenge_video_output.mp4 "Challenge Video Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in ".advanced_lane_finding.ipynb" in the function `calcCameraCalibrationPoints`.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted Image][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image1_1]

#### 2. Threshold image to find lines.

Thresholding is implemented in the function `thresholdImage`.
I used a combination of three different thresholds to detect lines in the image:
##### 1. Gradient in x direction and y direction or magnitude and direction
    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(gray, orient='x', thresh=(20, 100))
    grady = abs_sobel_thresh(gray, orient='y', thresh=(20, 100))
    
    # Magnitude and direction
    mag_binary = mag_thresh(gray, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray,sobel_kernel=15, thresh=(0.7, 1.3))
    
    # Combine sobelx, magnitude and direction
    sobel_combined = np.zeros_like(dir_binary)
    sobel_combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
##### 2. Color Threshold for white lines
    # Threshold white line
    r_thresh = (180, 255)
    g_thresh = (180, 255)
    b_thresh = (180, 255)
    wl_thres = (200, 255)
    wl_binary = np.zeros_like(R)
    wl_binary[(R > r_thresh[0]) & (R <= r_thresh[1]) & (G > g_thresh[0]) & (G <= g_thresh[1]) & 
               (B > b_thresh[0]) & (B <= b_thresh[1]) & (l_channel > wl_thres[0]) & (l_channel <= wl_thres[1])] = 1
##### 3. Color Threshold for yellow lines
    # Threshold yellow line
    ys_thresh = (100, 255)
    yh_thresh = (10, 80)
    yl_binary = np.zeros_like(R)
    yl_binary[(s_channel > ys_thresh[0]) & (s_channel <= ys_thresh[1]) & 
              (h_channel > yh_thresh[0]) & (h_channel <= yh_thresh[1])
             ] = 1
Finaly I combine all three parts together:
    # combined_binary[((wl_binary == 1) | (yl_binary==1)) | (sobel_combined == 1)] = 1

Here's an example of my output for this step.

![alt text][image1_2]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper`.  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
        [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
        [((img_size[0] / 6) + 5), img_size[1]],
       [(img_size[0] * 5 / 6) + 40, img_size[1]],
        [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
dst = np.float32(
        [[(img_size[0] / 4), 0],
        [(img_size[0] / 4), img_size[1]],
        [(img_size[0] * 3 / 4), img_size[1]],
        [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 218, 720      | 320, 720      |
| 1107, 720     | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Use sliding window and fit polynomial
In this step I get from the binary image to the two line markings that are described as polynomial. This step is implemented in the `fit_polynomial` function.
First I call the `find_lane_pixels` function. This function takes the binary thresholded image and creates a histogram of the bottom half of the image. Then I take the highest bars of the histogram on the left half and on the right half as start points for my left and right lane. Starting from the bottom of the image I slide up the image with a window for left and right lane. For every window the average position of white pixels is calculated. The average position is used as start point for the next window position.
Finaly I fit a second order polynomial through the sliding window positions:
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.
From the measure curvature II example we know that the lane width is 3.7 m and the length of a dashed line is 3 m. I measured the lane width and the line length in a warped image. This gives us a lane width of 640 pixel, and a line length of 100 pixel. We know stretch the stretch the warped image to by multiplying with norming factor in x and y dimension. Then I calculate the coefficients for the left and right lane from the stretched and warped image. With these coefficients we can now calculate the radius of the lanes in the real world.
To get the offset, I use the third coefficient of the polynomial and subtract the distance between the left positon of the image and the center of the image. This is done for left and right lane, added both together and devided the result by two. This gives us the offset to the center of the lane.
The described steps are implemented in the function `measure_curvature_real`.


#### 6. Plot result back down onto the road

Finally I transformed the lines back to the real world in the function `transformRealWorld`.
First, I create an empty image of the size of an warped image. Second, I draw my lanes into the image. Third, I calculate the invers matrix from the warped image to the original image. Fourth, I warp the image with the inverse matrix back. Last, I combine the warped back image with the original image. The result is shown in this picture:

![alt text][image6]

---

### Pipeline (video)

My pipeline for the video is implemented in function `ProcessImageR`.
![alt text][video1]
![alt text][video2]


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline seems to have issues if the lane markings have low contrast and if there are other lanes next to the lane marking. This could be improved by predicting the lanes and using the angle of steering. 
