import os
import sys
import cv2
import csv
from helpers import *


def calc_parameters(image, inverted_image):
    _, centroids, angles = count_sutures(inverted_image, image)

    inter_suture_distances = []
    for i in range(len(centroids) - 1):
        dist = np.linalg.norm(np.array(centroids[i]) - np.array(centroids[i + 1]))
        inter_suture_distances.append(dist)

    mean_spacing = np.mean(inter_suture_distances)
    variance_spacing = np.var(inter_suture_distances)
    
    mean_angle = np.mean(angles)
    variance_angle = np.var(angles)

    return mean_spacing, variance_spacing, mean_angle, variance_angle


def write_to_csv(output_csv, data):
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'number of sutures', 'mean inter suture spacing', 'variance of inter suture spacing', 'mean suture angle wrt x-axis', 'variance of suture angle wrt x-axis']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)


def part1(img_dir, output_csv):
    images_paths = open_files(img_dir)
    data = []

    for img_path in images_paths:
        img = cv2.imread(img_path)
        inverted_img = 255 - apply_threshold(apply_gray(apply_sharpen(img, apply_gaus(img))), 120)
        opened_img = apply_opening(inverted_img)

        # Counting sutures for each image
        num_sutures, _, _ = count_sutures(opened_img, img)

        # Other parameters
        mean_spacing, variance_spacing, mean_angle, variance_angle = calc_parameters(img, inverted_img)

        data.append({
            'image_name': os.path.basename(img_path),
            'number of sutures': num_sutures,
            'mean inter suture spacing': mean_spacing,
            'variance of inter suture spacing': variance_spacing,
            'mean suture angle wrt x-axis': mean_angle,
            'variance of suture angle wrt x-axis': variance_angle
        })

    write_to_csv(output_csv, data)
    print(f"[SUCCESS]CSV file '{output_csv}' populated with given data!")




if __name__ == "__main__":
    
    if len(sys.argv) < 4:
        print("Usage: python3 main.py <part_id> <img_dir> <output_csv>")
        sys.exit(1)

    part_id = int(sys.argv[1])
    img_dir = sys.argv[2]
    output_csv = sys.argv[3]

    if part_id == 1:
        part1(img_dir, output_csv)
    else:
        print("[ERROR]Invalid part id. Only part 1 working! :(")