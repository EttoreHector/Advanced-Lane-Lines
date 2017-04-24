# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 17:52:25 2017

@author: ettore
"""

import cv2
import numpy as np
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

from functions import (fit_polynomial, find_lane_lines_pixel,
                       calculate_curvature, plot_highlighted_lane_lines,
                       calculate_offset, find_calibration_parameters,
                       find_perspective_tranform_parameters, plot_images,
                       abs_sobel_thresh, hls_select, dir_thresh, project_back)

ym_per_pix = 3./72 # meters per pixel in y dimension
xm_per_pix = 3.7/680 # meters per pixel in x dimension


img_size = (1280,720)


    
##############################################################################
#### PIPELINE ####

# (This block of code is performed only once)

### 1. FIND DISTORTION PARAMETERS ###
ret, mtx, dist, rvecs, tvecs = find_calibration_parameters()

### 2. FIND PERSPECTIVE TRANSFORM PARAMETERS ###
M = find_perspective_tranform_parameters(img_size)
Minv = np.linalg.inv(M)

#-------------------------------------------------------
# The following steps of the pipeline are inside a function
# as they have to be performed for every image/frame.
def pipeline(img):
    
    ### 3. APPLY CAMERA CORRECTION ###
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    #plot_images(img, undist)
    
    ### 4. CREATE THRESHOLD BINARY IMAGE ###
    # (Sobel on x direction for R and B channel of the undistorted image)
    skernel = 3
    sobelx_mask_B = abs_sobel_thresh(undist[:,:,0], sobel_kernel=skernel, 
                                    orient='x', thresh_min=20,thresh_max=100)
    sobelx_mask_R = abs_sobel_thresh(undist[:,:,2], sobel_kernel=skernel, 
                                    orient='x', thresh_min=20,thresh_max=100)
    
    # (Direction of the gradient)
    dir_mask = dir_thresh(undist, sobel_kernel=skernel, thresh=(0.7, 1.3))
    
    # (Create binary mask in the hls color-space)   
    hls_mask = hls_select(undist, thresh=(120, 255))
    

    # (Create final binary mask)
    binary_mask = np.zeros_like(sobelx_mask_R)
    binary_mask[(((sobelx_mask_R == 1) | (sobelx_mask_B == 1)) | 
            (hls_mask == 1)) & (dir_mask == 1)] = 1
    #plot_images(undist, binary_mask, cm2='gray')
    #plt.imshow(binary_mask, cmap='gray')
    
                
    ### 5. APPLY PERSPECTIVE TRANSFORM ###
    warped = cv2.warpPerspective(binary_mask,M,img_size,flags=cv2.INTER_LINEAR) 
    #plot_images(binary_mask, warped, cm1='gray', cm2='gray')
    
    ### 6. FIND LANE LINE PIXELS ###
    leftx, lefty, rightx, righty = find_lane_lines_pixel(warped)
    
    ### 7. HIGHLIGHT LANE-LINES IN PICTURE ###
    left_fit_warped  = fit_polynomial(leftx,lefty)
    right_fit_warped = fit_polynomial(rightx,righty)
    #plot_highlighted_lane_lines(left_fit_warped,right_fit_warped,leftx,rightx,
    #                            lefty,righty,warped)
    
    ### 8. FIND LANE-LINES CURVATURE ###   
    left_fit_road  = fit_polynomial(leftx,lefty,x_conv_factor=xm_per_pix,
                                      y_conv_factor=ym_per_pix)
    right_fit_road = fit_polynomial(rightx,righty,x_conv_factor=xm_per_pix,
                                      y_conv_factor=ym_per_pix)
    left_curverad, right_curverad = calculate_curvature(left_fit_road,
                                                        right_fit_road,
                                                        img_size)
    
    ### 9. FIND CAR OFFSET WITH RESPECT TO THE MIDDLE LANE ###
    car_offset = calculate_offset(left_fit_warped,right_fit_warped,img_size,
                                  x_conv_factor=xm_per_pix)
    #print(left_curverad,'m - ', right_curverad,'m')
    #print(car_offset,' m')
    
    
    ### 10. COLOUR-UP LANE IN ORIGINAL UNDISTORTED IMAGE
    ploty = np.linspace(0, img_size[1]-1, num=img_size[1])
    left_fitx = (left_fit_warped[0]*ploty**2 + left_fit_warped[1]*ploty +
                 left_fit_warped[2])
    right_fitx = (right_fit_warped[0]*ploty**2 + right_fit_warped[1]*ploty + 
                  right_fit_warped[2])
    newimage = project_back(warped,undist,left_fitx,right_fitx,ploty,Minv)    
    #plt.imshow(newimage)

    return newimage

#-------------------------------------------

#pipeline(mpimg.imread('./test_images/test5.jpg'))

video_output = 'project_video_output.mp4'
clip = VideoFileClip('project_video.mp4')
new_clip = clip.fl_image(pipeline)
new_clip.write_videofile(video_output)
