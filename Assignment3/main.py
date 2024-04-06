import os
import numpy as np
import pickle
import ROI_Coordinates
import HoG
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import joblib
import cv2


train_folders = [
        "./dataset/closed/train_ROI",
        "./dataset/open/train_ROI"
    ]

validate_folders = [
        "./dataset/closed/valid_ROI",
        "./dataset/open/valid_ROI"
    ]


# 0 Closed, 1 Open
class_labels = {
    "./dataset/closed/train_ROI": 0,  
    "./dataset/closed/valid_ROI": 0,  
    "./dataset/open/train_ROI": 1,
    "./dataset/open/valid_ROI": 1      
}


def main():
    # Chk & perform preprocessing of Raw images (if required)
    ROI_Coordinates.process_ROI()
    print()

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
            label = class_labels[train_folder]
            image_filenames = os.listdir(train_folder)
            image_paths = [os.path.join(train_folder, filename) for filename in image_filenames]
            hoG_feat.extend([(feat, label) for feat in HoG.calculate_hog_features_multiple(image_paths)])

        print("[SUCCESS] HoG features computations for test images completed!")
        # Save HoG features to a file
        with open(hog_features_file, 'wb') as f:
            pickle.dump(hoG_feat, f)
        print(f"[LOG] Saved HoG features to Pickle dump!")

    # print(hoG_feat)
    print()

    
    model_file = 'svm_model.sav'
    if os.path.exists(model_file):
        print(f"[LOG] Loading existing SVM model from {model_file}!")
        svm_model = joblib.load(model_file)
    
    else:
        # Split features and labels
        X_train = np.array([feat for feat, _ in hoG_feat])
        y_train = np.array([label for _, label in hoG_feat])


        X_train_flattened = X_train.reshape(X_train.shape[0], -1)
        # print(X_train_flattened.shape)
        # print(y_train.shape)

        # Parameter grid for Grid Search (Hyper-parameter tuning)
        param_grid = {'C': [1, 10, 100], 'gamma': [0.01, 0.001], 'kernel': ['linear', 'rbf']}

        svm_model = svm.SVC(probability=True)
        grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=3, scoring='accuracy')
        
        print("[LOG] Training SVM model...")
        grid_search.fit(X_train_flattened, y_train)
        
        svm_model = grid_search.best_estimator_
        print("[SUCCESS] SVM model trained successfully.")
        print(grid_search.best_params_)
        
        joblib.dump(svm_model, 'svm_model.sav')
        print(f"[LOG] Saved SVM model to {model_file}")
    
    print()
    print()


    # Lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []
    predicted_probabilities = []

    output_directory = os.path.join(os.path.dirname(__file__), "validate_res")
    os.makedirs(output_directory, exist_ok=True)

    # Validation
    print("[*] ====================Model Statistics====================")
    for folder in validate_folders:
        print(f"[LOG] Evaluating model performance on {folder}")
        label = class_labels[folder]
        # print(f"[*] Label for {folder}: {label}")
        
        image_filenames = os.listdir(folder)
        image_paths = [os.path.join(folder, filename) for filename in image_filenames]
        hoG_features = HoG.calculate_hog_features_multiple(image_paths)
        
        X_test = np.array(hoG_features)
        X_test_flattened = X_test.reshape(X_test.shape[0], -1)
        y_pred = svm_model.predict(X_test_flattened)
        
        if not isinstance(label, np.ndarray):
            label = np.full_like(y_pred, label)

        true_labels.extend(label)
        predicted_labels.extend(y_pred)
        predicted_probabilities.extend(svm_model.predict_proba(X_test_flattened)[:, 1])

        accuracy = np.mean(y_pred == label) * 100
        print(f"[*] Accuracy on {folder}: {accuracy:.3f}%")
        

        for i, img_path in enumerate(image_paths):
            img = cv2.imread(img_path)
            if y_pred[i] == 0:
                predicted_label_text = "Predicted: Close"
            else:
                predicted_label_text = "Predicted: Open"

            cv2.putText(img, predicted_label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            output_img_path = os.path.join(output_directory, f"predicted_{y_pred[i]}_{os.path.basename(img_path)}")
            cv2.imwrite(output_img_path, img)
        print(f"[*] Saved images with predicted label at: {output_directory}")
        print()


    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)    
    predicted_probabilities = np.array(predicted_probabilities)


    overall_accuracy = np.mean(true_labels == predicted_labels) * 100
    overall_precision = precision_score(true_labels, predicted_labels)
    overall_recall = recall_score(true_labels, predicted_labels)
    overall_f1 = f1_score(true_labels, predicted_labels)

    print(f"[*] Overall Accuracy: {overall_accuracy:.3f}%")
    print(f"[*] Overall Precision: {overall_precision:.2f}")
    print(f"[*] Overall Recall: {overall_recall:.2f}")
    print(f"[*] Overall F1-score: {overall_f1:.2f}")

    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    roc_auc = auc(fpr, tpr)
    # print(f"[*] FPR : {fpr}")
    # print(f"[*] TPR : {tpr}")

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print(f"[*] ROC_curve is saved at the pwd as roc_curve.png!")



if __name__ == "__main__":
    main()


# To execute the script [WSL-Linux]: python3 main.py