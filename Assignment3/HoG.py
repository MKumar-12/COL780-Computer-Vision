from skimage import io, color
from skimage.transform import resize
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math


def preprocess_image(image_path):
    resized_img = resize(color.rgb2gray(io.imread(image_path)), (128, 64))
    grayscale_img = np.array(resized_img)
    return grayscale_img

def calculate_gradients(image):
    mag_gradients = []
    angle_gradients = []

    for i in range(128):
        magnitude_row = []
        angle_row = []

        for j in range(64):
            if j - 1 <= 0 or j + 1 >= 64:
                if j - 1 <= 0:
                    Gx = image[i][j + 1] - 0
                elif j + 1 >= len(image[0]):
                    Gx = 0 - image[i][j - 1]
            else:
                Gx = image[i][j + 1] - image[i][j - 1]
            
            if i - 1 <= 0 or i + 1 >= 128:
                if i - 1 <= 0:
                    Gy = 0 - image[i + 1][j]
                elif i + 1 >= 128:
                    Gy = image[i - 1][j] - 0
            else:
                Gy = image[i - 1][j] - image[i + 1][j]

            magnitude = math.sqrt(pow(Gx, 2) + pow(Gy, 2))
            magnitude_row.append(round(magnitude, 9))

            if Gx == 0:
                angle = math.degrees(0.0)
            else:
                angle = math.degrees(abs(math.atan(Gy / Gx)))
            
            angle_row.append(round(angle, 9))
        
        mag_gradients.append(magnitude_row)
        angle_gradients.append(angle_row)

    mag_gradients = np.array(mag_gradients)
    angle_gradients = np.array(angle_gradients)
    return mag_gradients, angle_gradients


def calculate_hog_features_multiple(image_paths):
    hog_features_list = []
    for image_path in tqdm(image_paths):
        hog_features = calculate_hog_features(image_path)
        hog_features_list.append(hog_features)
    return np.array(hog_features_list)


def calculate_hog_features(image_path):
    grayscale_img = preprocess_image(image_path)

    # Display initial img.
    # plt.figure(figsize=(15, 8))
    # plt.imshow(grayscale_img, cmap="gray")
    # plt.axis("off")
    # plt.savefig('init_grey_img.png')

    mag_gradients, angle_gradients = calculate_gradients(grayscale_img)

    # Magnitude of all blocks
    # plt.figure(figsize=(15, 8))
    # plt.imshow(mag_gradients, cmap="gray")
    # plt.axis("off")
    # plt.savefig('magnitude.png')

    # Dir. of all blocks
    # plt.figure(figsize=(15, 8))
    # plt.imshow(angle_gradients, cmap="gray")
    # plt.axis("off")
    # plt.savefig('dir.png')

    # Histogram Binning
    number_of_bins = 9
    step_size = 180 / number_of_bins

    def calculate_bin_index(angle):
        temp = (angle / step_size) - 0.5
        bin_index = math.floor(temp)
        return bin_index

    def calculate_bin_center(bin_index):
        bin_center = step_size * (bin_index + 0.5)
        return round(bin_center, 9)

    def calculate_bin_value(magnitude, angle, bin_index):
        bin_center = calculate_bin_center(bin_index + 1)
        bin_value = magnitude * ((bin_center - angle) / step_size)
        return round(bin_value, 9)

    histogram_points_nine = []
    for i in range(0, 128, 8):
        temp = []
        for j in range(0, 64, 8):
            magnitude_values = [[mag_gradients[i][x] for x in range(j, j+8)] for i in range(i,i+8)]
            angle_values = [[angle_gradients[i][x] for x in range(j, j+8)] for i in range(i, i+8)]
            for k in range(len(magnitude_values)):
                for l in range(len(magnitude_values[0])):
                    bins = [0.0 for _ in range(number_of_bins)]
                    value_j = calculate_bin_index(angle_values[k][l])
                    Vj = calculate_bin_value(magnitude_values[k][l], angle_values[k][l], value_j)
                    Vj_1 = magnitude_values[k][l] - Vj
                    bins[value_j]+=Vj
                    bins[value_j+1]+=Vj_1
                    bins = [round(x, 9) for x in bins]
            temp.append(bins)
        histogram_points_nine.append(temp)

    epsilon = 1e-05
    feature_vectors = []
    for i in range(0, len(histogram_points_nine) - 1, 1):
        temp = []
        for j in range(0, len(histogram_points_nine[0]) - 1, 1):
            values = [[histogram_points_nine[i][x] for x in range(j, j+2)] for i in range(i, i+2)]
            final_vector = []
            for k in values:
                for l in k:
                    for m in l:
                        final_vector.append(m)
            k = round(math.sqrt(sum([pow(x, 2) for x in final_vector])), 9)
            final_vector = [round(x/(k + epsilon), 9) for x in final_vector]
            temp.append(final_vector)
        feature_vectors.append(temp)

    # print(f'[LOG] Number of HOG features = {len(feature_vectors) * len(feature_vectors[0]) * len(feature_vectors[0][0])}')
    return feature_vectors