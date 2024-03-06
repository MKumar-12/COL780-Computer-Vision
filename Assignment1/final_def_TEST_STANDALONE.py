import os
import cv2
import numpy as np


# To Read all images from the given dir.
def open_files():
    # Storing data directory :
    main_dir = os.getcwd() 
    img_srcfolder = 'data'
    data_path = os.path.join(main_dir, img_srcfolder)

    #Storing the list of images :
    images_list = []
    
    if os.path.exists(data_path):
        for filename in os.listdir(data_path):
            img_path = os.path.join(data_path, filename)

            #read curr_img :
            img = cv2.imread(img_path)
            images_list.append(img)

    else:
        print(f"[ERROR]The '{img_srcfolder}' subfolder does not exist in the present directory!")
    
    return images_list


def apply_gaus(image):
    kernel_size = 3
    sigma = 1.0

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
    alpha = 1.5     #sharpening factor(alpha)
    beta = 0.5

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
    kernel_x5 = np.ones((1, 5), np.uint8)
    kernel_x10 = np.ones((1, 10), np.uint8)

    open_1 = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel_x5)
    open_2 = cv2.morphologyEx(inverted_image, cv2.MORPH_OPEN, kernel_x10)
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
            if len(contour) >= median_contour_length: 
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
    
    return ctr




# Taking i/p as images & storing them in a list as GrayScale
images_list = open_files()
print(f"[SUCCESS]Found {len(images_list)} image(s)...Ready to process!")

# Applying Gaussian Blur
blurred_images = [apply_gaus(img) for img in images_list]
print("\n[LOG]Applied Gaussion Blur to all images!")

# Sharpening Images
sharpened_images = [apply_sharpen(img, blur_img) for img, blur_img in zip(images_list, blurred_images)]
print("[LOG]Sharpened all images!")

# Converting to Single Channel : GrayScale
gray_images = [apply_gray(sharpen_img) for sharpen_img in sharpened_images]
print("[LOG]Converted all images to Single Channel!")

# Thresholding : for segmentation
threshold_val = 120
bin_images = [apply_threshold(gray_img, threshold_val) for gray_img in gray_images]
print("[LOG]Converted all images to Binary images!")

# Inversion of Images
inverted_images = [255 - bin_img for bin_img in bin_images]
print("\n[LOG]Inverted all images!")

# Applying opening to images (erosion followed by dilation)
opened_images = [apply_opening(inverted_img) for inverted_img in inverted_images]
print("[LOG]Applied Errosion & Dilation to -ve images!", end="\n")

# Counting sutures for each image
suture_counts = [count_sutures(opened_img, org_img) for opened_img, org_img in zip(opened_images, images_list)]
print("\n[SUCCESS]Following are the Sutures count : ", end = "\n")


# Displaying Results
for idx, (original_img, blurred_img, sharpen_img, gray_img, bin_img, inverted_img, opened_img, suture_count) in enumerate(zip(images_list, blurred_images, sharpened_images, gray_images, bin_images, inverted_images, opened_images, suture_counts)):
    # Display the images
    cv2.imshow(f'Original Image {idx + 1}', original_img)
    cv2.imshow(f'Blurred {idx + 1}', blurred_img)
    cv2.imshow(f'Sharpened {idx + 1}', sharpen_img)
    cv2.imshow(f'Gray {idx + 1}', gray_img)
    cv2.imshow(f'Threshholded {idx + 1}', bin_img)
    cv2.imshow(f'Inverted {idx + 1}', inverted_img)
    cv2.imshow(f'Opened Image {idx + 1}', opened_img)
    print(f'Image {idx + 1}: {suture_count}')
    cv2.waitKey(1000)


# Displaying a Specific img & related work :
# x = 3    
# cv2.imshow(f'Original Image', images_list[x])
# cv2.imshow(f'Blurred', blurred_images[x])
# cv2.imshow(f'Sharpened', sharpened_images[x])
# cv2.imshow(f'Gray', gray_images[x])
# cv2.imshow(f'Threshholded', bin_images[x])
# cv2.imshow(f'Inverted', inverted_images[x])
# cv2.imshow(f'Opened Image', opened_images[x])

if(cv2.waitKey() == 27) :
    cv2.destroyAllWindows()


'''
Img1 - 9
Img2 - 5
Img3 - 9
Img4 - 10
Img5 - 10           OK
Img6 - 13
Img7 - 8            OK
Img8 - 17
Img9 - 16
Img10 - 14

'''