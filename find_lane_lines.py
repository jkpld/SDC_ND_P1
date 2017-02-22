import numpy as np
import cv2
import math
from math import pi
from accumLines import *
from scipy.linalg import lstsq
from scipy.optimize import least_squares


class History:
    # This class will store the results from the past several image frames
    # and compute the average. Values stored are the voting landscape and
    # the end points for the left and right lane lines.

    def __init__(self, history_length, equal_weight=True):
        # Initialize storage for the current state of the voting landscape,
        # the left lane, and the right lane
        self.VL = np.zeros(0)  # np.zeros( img_size )
        self.left_lane = np.zeros((2, 2))  # [p_start; p_end]
        self.right_lane = np.zeros((2, 2))  # [p_start; p_end]

        # Initialize storage for the hisotry of values
        self.VL_hist = np.zeros(0)  # np.zeros( img_size + (history_length,) )
        self.left_lane_hist = np.zeros((2, 2) + (history_length,))
        self.right_lane_hist = np.zeros((2, 2) + (history_length,))

        if equal_weight:
            w = np.ones(history_length)
        else:
            w = np.linspace(0.3, 1, history_length)
        self.__weight = w  # .reshape((1,1,history_length))

        self.__history_length = history_length
        self._cntr = 0

        self.mask = []

    def add(self, VL, left_lane, right_lane):
        # Now that we know the size of the image we can initialize the
        # variables for the voting landscape
        if self._cntr == 0:
            img_size = VL.shape
            self.VL = np.zeros(img_size)
            self.VL_hist = np.zeros(img_size + (self.__history_length,))

        # Add to history
        self.VL_hist[:, :, 0] = VL
        self.VL_hist = np.roll(self.VL_hist, -1, axis=2)

        self.left_lane_hist[:, :, 0] = left_lane
        self.left_lane_hist = np.roll(self.left_lane_hist, -1, axis=2)

        self.right_lane_hist[:, :, 0] = right_lane
        self.right_lane_hist = np.roll(self.right_lane_hist, -1, axis=2)

        self._cntr += 1

        # Copmute new current values as mean of history
        if self._cntr < self.__history_length:
            W = np.sum(self.__weight[-self._cntr:])
            self.VL = self.VL_hist[
                :, :, -self._cntr:].dot(self.__weight[-self._cntr:]) / W
            self.left_lane = self.left_lane_hist[
                :, :, -self._cntr:].dot(self.__weight[-self._cntr:]) / W
            self.right_lane = self.right_lane_hist[
                :, :, -self._cntr:].dot(self.__weight[-self._cntr:]) / W
        else:
            W = np.sum(self.__weight)
            self.VL = self.VL_hist.dot(self.__weight) / W
            self.left_lane = self.left_lane_hist.dot(self.__weight) / W
            self.right_lane = self.right_lane_hist.dot(self.__weight) / W


def pre_process_image(I, roi, resize_factor=1):
    # Pre-process the images :
    # - resize them
    # - adjust contrast in ROI along rows
    # - smooth the image
    I = I.astype(np.float32)

    I = np.mean(I[:, :, :2], axis=2)

    # Resize image and ROI
    if resize_factor != 1:
        I = cv2.resize(I, (0, 0), fx=resize_factor, fy=resize_factor)
        roi = cv2.resize(roi.astype(np.float32), (0, 0),
                         fx=resize_factor, fy=resize_factor)
        roi = roi > 0

    # Adjust image contrast in ROI along rows
    tmpI = I.copy()
    # if np.issubsctype(I.dtype, np.integer):
    #     tmpI[np.logical_not(roi)] = np.iinfo(I.dtype).max
    # else:
    #     tmpI[np.logical_not(roi)] = np.Inf

    tmpI[np.logical_not(roi)] = np.Inf
    minI = np.percentile(tmpI, 30, axis=1, keepdims=True)
    I = I - minI  # Note, should probably smooth out minI with a moving average before subtracting
    I[I < 0] = 0

    tmpI = I.copy()
    tmpI[np.logical_not(roi)] = 1
    maxI = np.percentile(tmpI, 99, axis=1, keepdims=True)

    I = 255 * I / maxI
    I[I > 255] = 255

    # Smooth out the image with a guassian blur
    I = cv2.GaussianBlur(I, (7, 7), 1)  # This uses matlab symmetric padding

    return I, roi


def locate_perpendicular_lines(I, roi, threshold=2, reduce_points=True):

    # Get the size of the image
    img_size = I.shape

    # Image gradient along either direction
    Gx = cv2.Sobel(I, -1, 1, 0, ksize=3)
    Gy = cv2.Sobel(I, -1, 0, 1, ksize=3)

    G = np.sqrt(Gx * Gx + Gy * Gy)
    mask = G > np.max(G) / threshold

    if reduce_points:
        # We do not need quite so many points, so remove 2/3'rds of them.
        # This is not the same as just increasing the threshold on the mask
        # above
        mask2 = np.zeros(mask.size, dtype=bool)
        mask2[::3] = True
        mask2.shape = mask.T.shape
        mask2 = mask2.T

        mask = mask & mask2 & roi

    # Find the location of the non-zero pixels
    y_idx, x_idx = mask.nonzero()
    y_idx.shape = (y_idx.size, 1)
    x_idx.shape = (x_idx.size, 1)

    # Get the slope of the line at each of these points
    # -- the slope of the line is perpendicular to the gradient
    Th = np.arctan2(Gy[mask], Gx[mask])
    Th[Th < 0] += 2 * pi  # map [0,2*pi]
    Th = Th + pi / 2  # Perpendicular to gradient
    slope = np.tan(Th)  # slope of the lines
    slope.shape = (slope.size, 1)

    # Compute the end points of the lines that conver the image, have the
    # slopes as given by above and intersect the points above
    #
    # There are likely faster/better ways of doing this, but it works

    # Lines starting and ending at x-bounds (but exceeding y bounds)
    x = np.ones((slope.size, 2)) * np.array((1, img_size[1])) - 1
    y = slope * (x - x_idx) + y_idx

    # Lines starting and ending at y-bounds (but exceeding x bounds)
    yr = np.ones((slope.size, 2)) * np.array((1, img_size[0])) - 1
    xr = (yr - y_idx) / slope + x_idx

    # Sort xr/yr so that xr[:,0] is less than xr[:,1] (same order as x)
    sort_ind = xr.argsort(axis=1)
    xr = xr[np.arange(slope.size)[:, None], sort_ind]
    yr = yr[np.arange(slope.size)[:, None], sort_ind]

    # if y is ousize of the y-bounds, then replace row with xr/yr values
    # --start point
    replace = (y[:, 0] < 0) | (y[:, 0] > (img_size[0] - 1))
    x[replace, 0] = xr[replace, 0]
    y[replace, 0] = yr[replace, 0]

    # --end point
    replace = (y[:, 1] < 0) | (y[:, 1] > (img_size[0] - 1))
    x[replace, 1] = xr[replace, 1]
    y[replace, 1] = yr[replace, 1]

    # round all values to nearest integer and cast as integers
    x = np.around(x).astype(np.int_)
    y = np.around(y).astype(np.int_)

    # create arrays with the start point (y,x)_start and the end point
    # (y,x)_end (we do just convert back to the x,y in the accumLines
    # function, but, haveing starting/end points together seems nice.)
    x1 = np.column_stack((y[:, 0], x[:, 0]))
    x2 = np.column_stack((y[:, 1], x[:, 1]))

    return x1, x2


def compute_lines(vote_landscape, roi, previous_VL=None):

    # import matplotlib.pyplot as plt

    img_size = vote_landscape.shape

    VL = cv2.GaussianBlur(vote_landscape, (15, 15), 2)

    current_VL = VL.copy()

    if previous_VL is not None:
        if previous_VL.size != 0:
            VL = np.mean(np.dstack((VL, previous_VL)), axis=2)

    VL[np.logical_not(roi)] = 0

    VLsum0 = VL.sum(axis=0, keepdims=True)
    VLsum0[VLsum0 < 1] = 1

    VL = (VLsum0 > np.percentile(VLsum0, 25)) * VL / VLsum0
    VL = VL > np.percentile(VL, 97)

    y, x = VL.nonzero()
    y = y.astype(np.float32)
    x = x.astype(np.float32)

    x_mid = x.mean()

    y_min = np.around(img_size[0] * 0.6)
    # x0 = np.zeros(2)
    #
    # def fun(x, t, y):
    #     return x[1]*t + x[0] - y

    inrange = x > (x_mid + 10)
    X = np.column_stack((np.ones(np.count_nonzero(inrange)), x[inrange] + 1))

    # This returns the intercept and slope of the line
    wr = lstsq(X, y[inrange] + 1)[0]
    # wr = least_squares(fun, x0, args=(x[inrange], y[inrange]))
    # print(wr)

    # ends points of right line
    yr = np.array([img_size[0] - 1, y_min])
    xr = np.around((yr - wr[0]) / wr[1])

    inrange = x < (x_mid - 10)
    X = np.column_stack((np.ones(np.count_nonzero(inrange)), x[inrange] + 1))
    # This returns the intercept and slope of the line
    wl = lstsq(X, y[inrange] + 1)[0]
    # wl = least_squares(fun, x0, args=(x[inrange], y[inrange]))[0]
    # ends points of right line
    yl = np.array([img_size[0] - 1, y_min])
    xl = np.around((yl - wl[0]) / wl[1])

    ll = np.column_stack((yl, xl))  # left lane [p1; p2]
    rl = np.column_stack((yr, xr))  # right lane [p1; p2]

    return ll, rl, current_VL, VL


def create_ROI(img_size, vertices):
    # Vertices should be an array of x,y values that give the fraction
    # the image.
    roi = np.zeros(img_size)
    vrts = np.around(
        vertices * (img_size[1] - 1, img_size[0] - 1)).astype(np.int32)
    cv2.fillPoly(roi, vrts, 1)
    roi = roi > 0
    return roi


def find_frames_lane_lines(I, roi, history):

    # Get image size
    orig_img_size = I.shape[0:2]

    # Define some constants
    # The imageas do need to be so large, so scale them down to this size if
    # larger
    WORKING_IMAGE_WIDTH = 400
    RESIZE_FACTOR = np.around(WORKING_IMAGE_WIDTH / orig_img_size[1], 2)
    # RESIZED_IMAGE = (int(np.around(orig_img_size[1]*RESIZE_FACTOR)), int(400))
    # Resize image, adjust contrast in ROI along rows, and smooth out the
    # image with a gaussian blur (sig=2)
    I, roi = pre_process_image(I, roi, RESIZE_FACTOR)

    # Get image size
    img_size = I.shape[0:2]

    # Find the end points for the lines perpendicular to large
    # gradients of I in ROI
    x1, x2 = locate_perpendicular_lines(I, roi)

    # For each of the lines computed above, accumulate the pixels they
    # cross over in an image, this will make all straight lines much easier
    # to threshold and the dotted lines will be automatically connected.
    A = accumLines(x1, x2, img_size)

    # Compute the end points of the lines and return the processed
    # accumulated matrix for saving
    ll, rl, VL, mask = compute_lines(A, roi, history.VL)

    # Rescale the computed lines to the original image size
    ll = ll / RESIZE_FACTOR
    rl = rl / RESIZE_FACTOR

    # Add new observation to history
    history.add(VL, ll, rl)
    # saving the mask only because it will be helpful to plot while describing
    # the method
    history.mask = mask
