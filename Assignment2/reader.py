#reader.py

import os
import cv2


# To Read all images from the given dir.
def open_image_files(img_dir):
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
                print(f"[WARNING]Skipping non-string filename: {filename}")

    else:
        print(f"[ERROR]The '{img_dir}' subfolder does not exist in the present directory!")
    
    return image_paths


# To Read video from the given dir. & generate screenshots from it every 1s
def open_video_file(video_dir):
    main_dir = os.getcwd() 
    data_path = os.path.join(main_dir, video_dir)

    #Storing the list of extracted images :
    image_paths = []

    if os.path.exists(data_path):
        # Open the video file
        cap = cv2.VideoCapture(data_path)
        if not cap.isOpened():
            print(f"[ERROR] Could not open video file: {data_path}")
            return image_paths

        # Create a folder 'src' to store frames captured from video
        gen_dir = os.path.join(main_dir, "src")
        if not os.path.exists(gen_dir):
            os.makedirs(gen_dir)

        # Extract frames at 1-second intervals 
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(frame_rate * 1)
        frame_ctr = 0
        success, image = cap.read()
        while success:
            image_paths.append(os.path.join(gen_dir, f"frame_{frame_ctr}.jpg"))
            cv2.imwrite(os.path.join(gen_dir, f"frame_{frame_ctr}.jpg"), image)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_ctr + frame_interval)
            success, image = cap.read()
            frame_ctr += frame_interval

        cap.release()

    else:
        print(f"[ERROR]The '{video_dir}' subfolder does not exist in the present directory!")    

    return image_paths