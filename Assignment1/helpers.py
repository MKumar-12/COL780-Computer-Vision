#helper.py

import os
import cv2
import numpy as np


# To Read all images from the given dir.
def open_files(img_dir):
    # Storing data directory :
    main_dir = os.getcwd() 
    data_path = os.path.join(main_dir, img_dir)

    #Storing the list of images :
    image_paths = []
    
    if os.path.exists(data_path):
        for filename in os.listdir(data_path):
            if isinstance(filename, str):  # Ensure filename is a string
                img_path = os.path.join(data_path, filename)
                image_paths.append(img_path)
            else:
                print(f"[WARNING] Skipping non-string filename: {filename}")

    else:
        print(f"[ERROR]The '{img_dir}' subfolder does not exist in the present directory!")
    
    return image_paths


def apply_gaus(image):
    kernel_size = 3
    sigma = 2.0

    blurred_channels = [gaussian_blur_single_channel(channel, kernel_size, sigma) for channel in cv2.split(image)]
    blurred_image = cv2.merge(blurred_channels)
    return blurred_image


def gaussian_blur_single_channel(channel, kernel_size, sigma):
    kernel = gaussian_kernel(kernel_size, sigma)

    blurred_channel = convolution(channel, kernel)      # Applying convolution
    return blurred_channel


def gaussian_kernel(size, sigma):
    if sigma == 0:
        # No blurring, return a kernel with all zeros except for the center
        kernel = np.zeros((size, size))
        kernel[size // 2, size // 2] = 1
        return kernel
    else:
        kernel = np.fromfunction(
            lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
            (size, size)
        )
        return kernel / np.sum(kernel)


def convolution(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape

    pad_height = k_height // 2
    pad_width = k_width // 2

    # Create zero-padded image
    padded_image = np.zeros((height + 2 * pad_height, width + 2 * pad_width))
    padded_image[pad_height:pad_height+height, pad_width:pad_width+width] = image

    # Initialize result image
    result_image = np.zeros_like(image)

    # Perform convolution
    for i in range(height):
        for j in range(width):
            result_image[i, j] = np.sum(padded_image[i:i+k_height, j:j+k_width] * kernel)

    return result_image.astype(np.uint8)


def apply_sharpen(image, blurred_image):
    # subtracting a blurred version of img. from the original img.
    alpha = 4.5     #sharpening factor(alpha)
    beta = 3.5

    sharpened_image = (alpha * image) - (beta * blurred_image)
    sharpened_image = np.clip(sharpened_image, 0, 255).astype(np.uint8)
    return sharpened_image


def apply_gray(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    return gray_img              


def apply_threshold(image, threshold_value):
    thresholded_image = np.where(image >= threshold_value, 255, 0).astype(np.uint8)
    return thresholded_image


def apply_opening(inverted_image):
    kernel_x12 = np.ones((1, 12), np.uint8)
    kernel_x15 = np.ones((1, 15), np.uint8)

    open_1 = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel_x12)
    open_2 = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel_x15)
    opened_image = cv2.bitwise_and(open_1, open_2)
    return opened_image


def count_sutures(binary_image, org_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    ctr = 0
    total_contour_len = 0
    contour_len = []
    centroids = []
    angles = []

    for contour in contours:
        length = len(contour)
        contour_len.append(length)
        total_contour_len += length

    if len(contours) > 0:
        average_contour_length = total_contour_len / len(contours)
        median_contour_length = np.median(contour_len)
        # print("\n[LOG]Average contour length:", average_contour_length)
        # print("[LOG]Median contour length:", median_contour_length)

        for contour in contours:
            # print(len(contour))
            if len(contour) >= (average_contour_length - 20): 
                ctr += 1
                leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
                rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
                centroid = tuple(contour.mean(axis=0)[0].astype(int))
                centroids.append(centroid)

                angle = np.arctan2(leftmost[1] - centroid[1], leftmost[0] - centroid[0])
                angles.append(angle)

                cv2.circle(org_image, leftmost, 4, (100, 255, 0), -1)
                cv2.circle(org_image, rightmost, 4, (140, 155, 270), -1)
                cv2.circle(org_image, centroid, 6, (255, 255, 0), -1)

        # print("[LOG]Contours with len. greater than median contour len: ", ctr)
    
    return ctr, centroids, angles