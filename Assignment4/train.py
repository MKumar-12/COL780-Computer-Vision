import os
import cv2


# Training RCNN model using YOLO dataset    
train_dir = "./mammo_1k/yolo_1k/train"
validate_dir = "./mammo_1k/yolo_1k/val"
    

# checks if each folder contains images and labels subfolders
def check_folder_structure(data_dir):
    sub_folders = ["images", "labels"]
    
    for sub in sub_folders:
        if not os.path.isdir(os.path.join(data_dir, sub)):
            return False
    
    return True


# parse YOLO format annotations from text files
def parse_yolo_annotations(annotation_file):
    class_label = 0 if os.path.isfile(annotation_file) else 1
    # For images without annotations (benign), label -> 1

    if class_label == 0:
        with open(annotation_file, 'r') as file:  # (label, x, y, w, h)
            line = file.readline().strip()
            parts = line.split()
            class_label = int(parts[0])  # Object class label -> 0 : tumor present
            x_center = float(parts[1])   # Normalized x-coordinate of the center of the bounding box
            y_center = float(parts[2])   # Normalized y-coordinate of the center of the bounding box
            width = float(parts[3])      # Normalized width of the bounding box
            height = float(parts[4])     # Normalized height of the bounding box
        return class_label, x_center, y_center, width, height
    else:
        return class_label, None, None, None, None      # Bounding box not present


# load images and annotations from the dataset
def load_dataset(data_dir):
    image_dir = os.path.join(data_dir, "images")
    label_dir = os.path.join(data_dir, "labels")
    dataset = []
    
    for filename in os.listdir(image_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + ".txt")
            
            annotations = parse_yolo_annotations(label_path)
            dataset.append((image_path, annotations))
    
    return dataset


def main():
    if check_folder_structure(train_dir):
        print("[LOG] Folder structure is correct as per i/p YOLO Dataset!")
        
        dataset = load_dataset(train_dir)
        print(f"[DEBUG] #Images found in train dir. : {len(dataset)}")
        print(f"[DEBUG] #Malignant Images found in train dir. : {sum(1 for item in dataset if item[1][0] == 0)}")
        # remaining images in the dir. are benign images (do not have tumors)
        print(f"[DEBUG] #Benign Images found in train dir.    : {sum(1 for item in dataset if item[1][0] == 1)}")
        
        print("\n\n[LOG] Sample annotations:")
        for image_path, annotations in dataset[:3]:
            print("Image:", image_path)
            print("Annotations:", annotations)
            
            if annotations[0] == 0:  # tumor present
                image = cv2.imread(image_path)
                h, w, _ = image.shape 
                x_center = int(annotations[1] * w)
                y_center = int(annotations[2] * h)
                box_width = int(annotations[3] * w)
                box_height = int(annotations[4] * h)
                x1 = int(x_center - box_width/2)
                y1 = int(y_center - box_height/2)
                x2 = int(x_center + box_width/2)
                y2 = int(y_center + box_height/2)
                cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)  # plot bounding box
                cv2.putText(image, 'proximate_tumor_area', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2) 
                cv2.imshow(f"{os.path.splitext(os.path.basename(image_path))[0]}", image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                print()
            else:
                print("No tumor present\n")
    else:
        print("[ERROR] Folder structure is incorrect. Make sure the folder contains 'images' and 'labels' subfolders.")


if __name__ == "__main__":
    main()