import os
import sys
import cv2
import csv
from helpers import *

def write_to_csv(output_csv, data):
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'number of sutures', 'mean inter suture spacing', 'variance of inter suture spacing', 'mean suture angle wrt x-axis', 'variance of suture angle wrt x-axis']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerows(data)


def part1(img_dir, output_csv):
    
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