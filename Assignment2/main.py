import os
import sys
from reader import *
from helpers import *

# Creation of Panaroma from given set of images in a DIR
def part1(input_path, output_path):
    image_paths = open_image_files(input_path)
    Images = []

    if not image_paths:
        print(f"[ERROR] No images found in '{input_path}'.")
        return
    
    print(f"Found {len(image_paths)} images in the given Directory!")
    
    for img_path in image_paths:
        img = cv2.imread(img_path)
        Images.append(img)

    BaseImage, _, _ = ProjectOntoCylinder(Images[0])   
    for i in range(1, len(Images)):
        StitchedImage = StitchImages(BaseImage, Images[i])
        print(i)
        BaseImage = StitchedImage.copy()    

    cv2.imwrite("Stitched_Panorama.png", BaseImage) 
    print(f"[SUCCESS] Panaromic image generated at location : '{output_path}'")



# Creation of Panaroma from given video in a DIR
def part2(input_path, output_path):
    src_paths = open_video_file(input_path)

    if not src_paths:
        print(f"[ERROR] Couldn't extract any image from video in '{input_path}'.")
        return
    
    # print("[INFO] List of extracted image paths:")
    # for img_path in src_paths:
    #     print(img_path)

    print(f"[SUCCESS] Panaromic image generated at location : '{output_path}'")


if __name__ == "__main__":   
    if len(sys.argv) != 4:
        print("Usage: python main.py <part_id> <img_dir> <output_dir>")
        sys.exit(1)

    part_id = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    if not os.path.exists(input_path):
        print(f"[ERROR] Input path '{input_path}' does not exist.")
        sys.exit(1)

    if part_id == 1:
        part1(input_path, output_path)
    elif part_id == 2:
        part2(input_path, output_path)
    else:
        print("[ERROR] Invalid part id. Must be either 1 or 2...")