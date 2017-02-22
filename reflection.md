# **Finding Lane Lines on the Road**
---

**Overview**

The goals / steps of this project are to make a pipeline that finds lane lines on the road and to reflect on the work in a written report

**Summary**

In this project I do not use either Canny edge detection or the Hough transform. Directly using the image gradient, I accumulate a set of lines along the image edges. From this accumulation matrix I compute the best left and right lane lines (in a least squared sense).

**Results**

The method implemented sucussfully located the lane lines in all three test videos. The main drawback was that the computation speed was slower than the videos' frame rates.

[//]: # (Image References)

[image0]: ./test_images/processed_solidWhiteCurve.jpg "Example with lines computed"
[image1]: ./preProcessed_image.jpg "Pre-processed image"
[image2]: ./voteLandscape.jpg "Voting landscape"
[image3]: ./voteLandscapeMask_withLines.jpg "Masked voting landscape with the calculated lane lines"

*Example frame of the results:*
![example_results][image0]

---

## Reflection

### 1. Describe your pipeline

In this project, I used different methods than those described in the text. The work flow is as follows:

#### Pre-process the image
* Create a gray-scale image by taking the mean of the red and green channel. Do not include the blue channel - this will make yellow and white the same.
* Adjust the image contrast along the rows inside the region of interest (ROI)
* Smooth out the image with a gaussian filter

![preprocess][image1]

#### Intermediate step : create voting landscape
* Compute the gradient (magnitude and direction) of the pre-processed image. The direction of the gradient is perpendicular to the edges of the image.
* Take a point with a large gradient, draw a line through that point with a direction perpendicular to the gradient. Write this line to a blank image.
* Repeat this process for each point with a large gradient and accumulate the values in the blank image. The resulting image is shown below.

![VL][image2]

#### Compute mask and fit lines
* Smooth the voting landscape (*VL*) with a gaussian filter, set everything outside the ROI to zero, and normalize each column of *VL* to have the same sum. This normalization works to equalize the intensities created from dotted lines and solid lines.
* Threshold the resulting *VL* and use the mean x value of the mask to divide the image into left and right parts
* For each of the parts, fit the pixels of the mask with a linear line

![VLmask_lines][image3]

#### Implement history

To gain robustness, I save a history of the results from the last several image frames. In particular, the voting landscape (*VL*), the end points of the left lane (*LL*), and the end points of the right lane (*RL*) are saved.

After processing a frame, the *VL*, *LL*, and *RL* are pushed into the history. The lane lines drawn on that frame are given by the mean of all *LL* and *RL* entries in the history. Additionally, the mean *VL* of the history is used to stabilize the *VL* of the next frame, before computing the *LL* and *RL* of that frame.

The end result of the history, is that the lines are smooth, and one or two frames can pass without finding the lane lines and still the results are stable.

**More details can be found in the** *ipython notebook* **and the files** *find_lane_lines.py* **and** *accumLines.py*

### 2. Identify potential shortcomings with your current pipeline

**Speed** *(in Python)* : The computation time results in a frame rate of about 15 frames per second (FPS). According to the video, the images were taken at 25 FPS, which means this could not be used in real time. *However*, I first implemented all of these methods in Matlab (as I am quite new in python), and my Matlab implementation achieved 30-40 FPS for all three of the test videos.

**More curved roads** : I do not know if it will work well on more curved roads. The reason is from making the lines in the accumulation step span the entire image. Thus, if a road is more curved, the voting landscape will *fan* outwards.

**Lane line extraction** : The process of extracting the lane lines from the *VL* mask should be improved. It works fine on the simple tests for this project, but when the mask is not symmetric or has noise it will fail.

### 3. Suggest possible improvements to your pipeline

A method for improving the lane line extraction could be to use the Hough transform on the *VL* mask. I implemented this in tests and it works will and improves the results on more curved roads; however, it results in a speed drop.

Another method to improve the extraction would be so used a robust linear fit when extracting the lane lines. Currently, a least sum of squares method is used to fit the lines, and while fast, it can suffer quite drastically from some noise in the *VL* mask.
