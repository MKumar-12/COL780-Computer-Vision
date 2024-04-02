import os
import numpy as np
import pickle
import ROI_Coordinates
import HoG

train_folders = [
        "./dataset/closed/train_ROI",
        "./dataset/open/train_ROI"
    ]

validate_folders = [
        "./dataset/closed/valid_ROI",
        "./dataset/open/valid_ROI"
    ]


def main():
    # Chk & perform preprocessing of Raw images (if required)
    ROI_Coordinates.process_ROI()

    # Check if the HoG features file already exists
    hog_features_file = 'hog_features.pkl'
    if os.path.exists(hog_features_file):
        print(f"[LOG] Loaded HoG features from existing Pickle dump!")
        # Load HoG features from the file
        with open(hog_features_file, 'rb') as f:
            hoG_feat = pickle.load(f)
    
    else:
        hoG_feat = []

        for train_folder in train_folders:
            print(f"[LOG] Extracting HoG features from {train_folder} directory...")
            image_filenames = os.listdir(train_folder)
            image_paths = [os.path.join(train_folder, filename) for filename in image_filenames]
            hoG_feat.append(HoG.calculate_hog_features_multiple(image_paths))

        # Save HoG features to a file
        with open(hog_features_file, 'wb') as f:
            pickle.dump(hoG_feat, f)
        print(f"[LOG] Saved HoG features to Pickle dump!")

    # print(hoG_feat)
    print("[SUCCESS] HoG features computations for test images completed!")

    X_train = np.concatenate(hoG_feat)
    print(X_train.shape)

    X_train_flattened = X_train.reshape(X_train.shape[0], -1)
    print(X_train_flattened.shape)


if __name__ == "__main__":
    main()
