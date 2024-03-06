import os
import cv2
import numpy as np


# Function to apply opening (erosion followed by dilation)
def apply_opening(binary_image, kernel_size=4):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    return opened_image


# Function to apply the Sobel operator for edge detection
def sobel_edge_detection(image):
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])            # Sobel kernel_x
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])            # Sobel kernel_y

    # Applying Sobel filters
    rows, cols = image.shape
    gradient_x = np.zeros_like(image, dtype=np.float32)
    gradient_y = np.zeros_like(image, dtype=np.float32)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            gradient_x[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * sobel_x)
            gradient_y[i, j] = np.sum(image[i-1:i+2, j-1:j+2] * sobel_y)


    # Magnitude of the gradients
    gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)

    # Normalizing the gradient magnitude to range [0, 255]:
    gradient_mag_normalized = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)

    return gradient_x, gradient_y, gradient_mag_normalized


# Function to apply non-maximum suppression(NMS) :
def non_maximum_suppression(gradient_mag, gradient_dir):
    rows, cols = gradient_mag.shape
    result = np.zeros_like(gradient_mag)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = gradient_dir[i, j]

            # Determine the neighbor pixels based on the gradient direction
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                neighbors = [gradient_mag[i, j + 1], gradient_mag[i, j - 1]]
            elif 22.5 <= angle < 67.5:
                neighbors = [gradient_mag[i - 1, j - 1], gradient_mag[i + 1, j + 1]]
            elif 67.5 <= angle < 112.5:
                neighbors = [gradient_mag[i - 1, j], gradient_mag[i + 1, j]]
            elif 112.5 <= angle < 157.5:
                neighbors = [gradient_mag[i - 1, j + 1], gradient_mag[i + 1, j - 1]]

            # Perform non-maximum suppression
            if gradient_mag[i, j] >= max(neighbors):
                result[i, j] = gradient_mag[i, j]

    return result


# Function to apply hysteresis thresholding
def hysteresis_thresholding(image, low_threshold, high_threshold):
    weak = 50  # Intensity value for weak edges
    strong = 255  # Intensity value for strong edges

    # Identify strong and weak edges based on thresholding
    strong_edges = (image >= high_threshold).astype(np.uint8) * strong
    weak_edges = ((image >= low_threshold) & (image < high_threshold)).astype(np.uint8) * weak

    # Perform hysteresis thresholding
    _, labels, stats, _ = cv2.connectedComponentsWithStats((strong_edges + weak_edges).astype(np.uint8), connectivity=8)

    # Assign strong intensity to all connected components labeled as strong
    for i in range(1, stats.shape[0]):
        if stats[i, cv2.CC_STAT_AREA] > 0:
            strong_edges[labels == i] = strong

    return strong_edges


# Inversion of img :
def inversion(image):
    # Set a threshold value (adjust as needed)
    threshold_value = 128

    # Convert the image to binary based on the threshold
    _, binary_image = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)

    # Invert the binary image
    inverted_binary_image = cv2.bitwise_not(binary_image)

    return inverted_binary_image


# Counting sutures
def count_sutures(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    count = 0
    total_contour_length = 0
    contour_lengths = []

    for cnt in contours:
        length = len(cnt)
        total_contour_length += length
        contour_lengths.append(length)

    if len(contours) > 0:
        average_contour_length = total_contour_length / len(contours)
        median_contour_length = np.median(contour_lengths)
        print("Average contour length:", average_contour_length)
        print("Median contour length:", median_contour_length)

        for cnt in contours:
            if len(cnt) > 100: 
                count += 1
                leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
                rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
                cv2.circle(binary_image, leftmost, 4, (100, 255, 0), -1)
                cv2.circle(binary_image, rightmost, 4, (140, 155, 270), -1)

        print("Image has threads with length greater than the median contour length:", count)
    
    return count




# Storing data directory :
main_dir = os.getcwd() 
img_srcfolder = 'data'
data_path = os.path.join(main_dir, img_srcfolder)

if os.path.exists(data_path):
    #Storing the list of images :
    images_list = []

    for filename in os.listdir(data_path):
        img_path = os.path.join(data_path, filename)

        #read curr_img :
        img = cv2.imread(img_path)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        images_list.append(gray_img)                            # storing single-channel images to reduce computations (color-information is not reqd. currently)

else:
    print(f"[ERROR]The '{img_srcfolder}' subfolder does not exist in the specified directory!")




# Applying opening to the edge-detected images
# opened_images = [apply_opening(edge_img) for edge_img in edge_detected_images]
opened_images = [apply_opening(img) for img in images_list]


# Edge detection using Sobel operator :
edge_detected_images = [sobel_edge_detection(opened_image) for opened_image in opened_images]


# Applying non-maximum suppression
nms_images = [non_maximum_suppression(gradient_mag, np.arctan2(gradient_y, gradient_x))
              for (gradient_x, gradient_y, gradient_mag) in edge_detected_images]


# Applying hysteresis thresholding
hysteresis_images = [hysteresis_thresholding(nms_img, low_threshold=60, high_threshold=120) for nms_img in nms_images]


# Inverting images processed by NMS :
# inverted_images = [inversion(nms_img) for nms_img in nms_images]


# Counting sutures for each image
suture_counts = [count_sutures(hysteresis_img) for hysteresis_img in hysteresis_images]
# suture_counts = [count_sutures(inverted_img) for inverted_img in inverted_images]
# suture_counts = [count_sutures(nms_img) for nms_img in nms_images]

# display fn. for all images : 
for idx, (original_img, opened_img, edge_detected_img, nms_img, hysteresis_img, suture_count) in enumerate(zip(images_list, opened_images, edge_detected_images, nms_images, hysteresis_images, suture_counts)):
    # Display the images
    # cv2.imshow(f'Original Image {idx + 1}', original_img)
    # cv2.imshow(f'Opened Image {idx + 1}', opened_img)
    # cv2.imshow(f'Edge Detected Image {idx + 1}', edge_detected_img[2]) 
    # cv2.namedWindow(f'NMS Image {idx + 1}', cv2.WINDOW_GUI_EXPANDED)
    # cv2.imshow(f'NMS Image {idx + 1}', nms_img)
    # cv2.imshow(f'Inverted Image {idx + 1}', inverted_img)
    cv2.namedWindow(f'Hysteresis Image {idx + 1}', cv2.WINDOW_NORMAL)
    cv2.imshow(f'Hysteresis Image {idx + 1}', hysteresis_img)
    # cv2.setMouseCallback(f'Hysteresis Image {idx + 1}', lambda event, x, y, flags, param: show_pixel_value(event, x, y, flags, param, hysteresis_img))
    print(f'Number of Sutures in Image {idx + 1}: {suture_count}')
    cv2.waitKey(1000)

# cv2.imshow(f'Original Image 4', images_list[3])
# cv2.imshow(f'Opened Image 4', opened_images[3])
# cv2.imshow(f'Edge Detected Image 4', edge_detected_images[3][2]) 
# cv2.imshow(f'NMS Image 4', nms_images[3])
# cv2.imshow(f'Hysteresis Image 4', hysteresis_images[3])
# print(f'Number of Sutures in Image 4: {suture_counts[3]}')

if(cv2.waitKey() == 27) :
    cv2.destroyAllWindows()


#use of NAMED_WINDOW , WINDOW_NORMAL -> enable resizing
    

'''
Img1 - 9
Img2 - 5
Img3 - 9
Img4 - 10
Img5 - 10           
Img6 - 13
Img7 - 8            
Img8 - 17
Img9 - 16
Img10 - 14

'''