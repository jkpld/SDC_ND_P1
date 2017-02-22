import numpy as np
from numpy import zeros, ones, absolute, around

def _pixels_in_range(p1,p2):
    # Helper function to create a list of pixels
    if p2 > p1:
        ps = np.arange(p1,p2+1)
    else:
        ps = np.arange(p1,p2-1,-1)

    return ps

def accumLines(x1,x2,img_size):
# ACCUMLINES accumulate lines in an image of size img_size given their end
# points x1, x2.
#
# In addition to just accumulating the lines, we also weight the lines
# using their slope. Horizontal lines get a weight of zero. Lines with a
# larger dx than dy get weighted by (dy/dx).
#
# x1 : n x 2 array, [y1, x1; ... ; yn, xn], of line start points
# x2 : n x 2 array, [y1, x1; ... ; yn, xn], of line end points
# img_size : 1x2 array with the size of the image
# conn : line connectivity --- only use 4 in this function

# Trace a line through a 2d grid and find each pixel the line intersects
# http://playtechs.blogspot.com/2007/03/raytracing-on-grid.html
# http://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator

    # If x1 is 1-d, then make sure it is treated as a column vector
    if np.ndim(x1) == 1:
        x1.shape = (1,2)
        x2.shape = (1,2)

    # Number of lines to accumulate
    N = x1.shape[0]
    A = zeros(img_size)

    # Get the differences for each line
    dy = x1[:,0] - x2[:,0]
    dx = x1[:,1] - x2[:,1]

    # Create an array of just the y and x values
    y = np.column_stack( (x1[:,0], x2[:,0]) )
    x = np.column_stack( (x1[:,1], x2[:,1]) )

    for pt in range(N):

        if dy[pt] == 0: # horizontal line

            # Create the pixels of the line
            xs = _pixels_in_range(x[pt,1],x[pt,0])
            ys = y[pt,0]*ones(xs.size)

            # Create the weights for the line
            w = zeros(xs.size)

        elif dx[pt] == 0: # vertical line

            # Create the pixels of the line
            ys = _pixels_in_range(y[pt,1],y[pt,0])
            xs = x[pt,0]*ones(ys.size)

            # Create the weights for the line
            w = ones(ys.size)

        else: # general line
            if absolute(dy[pt]) > absolute(dx[pt]):
                slope = dx[pt] / dy[pt]

                # Create the pixels of the line
                ys = _pixels_in_range(y[pt,1],y[pt,0])
                xs = around(slope * (ys - y[pt,0])) + x[pt,0]

                # Create the weights for the line
                w = ones(ys.size)
            else:
                slope = dy[pt] / dx[pt]

                # Create the pixels of the line
                xs = _pixels_in_range(x[pt,1],x[pt,0])
                ys = around(slope * (xs - x[pt,0])) + y[pt,0]

                # Create the weights for the line
                w = absolute(slope) * ones(ys.size)

        # Incriment our matrix
        A[ys.astype(np.int_),xs.astype(np.int_)] += w

    return A


#########################################################################
# This is code for connectivity 8, it is just in Matlab language now
# though. Connectivity 4 seems to work fine for this problem and is a bit
# faster.
#
# for pt = 1:N
#
#     dy = abs(x1(pt,1)-x2(pt,1));
#     dx = abs(x1(pt,2)-x2(pt,2));
#     y = x1(pt,1);
#     x = x1(pt,2);
#     n = 1 + dy + dx;
#     if x1(pt,1) > x2(pt,1)
#         y_inc = -1;
#     else
#         y_inc = 1;
#     end
#
#     if x1(pt,2) > x2(pt,2)
#         x_inc = -1;
#     else
#         x_inc = 1;
#     end
#
#     err = dy - dx;
#
#     dy = 2*dy; % Multipling by 2 makes everything in this problem an integer.
#     dx = 2*dx;
#
#     inds = zeros(n,1); % linear index
#
#     i = 1;
#     while i < n+1
#         inds(i) = y + (x-1)*img_size(1);
#
#         if err > 0
#             y = y + y_inc;
#             err = err - dx;
#         else
#             x = x + x_inc;
#             err = err + dy;
#         end
#         i = i + 1;
#     end
#
#     A(inds) = A(inds) + 1;
# end
#########################################################################
