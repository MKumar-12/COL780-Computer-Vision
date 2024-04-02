import cv2
import numpy as np
import glob
import pickle


# Load SVM Classifier
filename = open("SVM_Classifier_Hand.sav", 'rb')
clf = pickle.load(filename)

# Function to extract hand regions and classify them
def detect_and_classify_hands(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use a hand detection algorithm to identify potential hand regions
    # Example: Background Subtraction, Haar Cascades, etc.
    # Here, you can implement the hand detection algorithm of your choice
    
    # Extract hand regions based on the detection algorithm
    # For now, let's assume hand regions are already obtained
    
    # Iterate through detected hand regions
    for hand_region in detected_hand_regions:
        # Extract ROI from the hand region
        x, y, w, h = hand_region
        hand_roi = gray[y:y+h, x:x+w]
        
        # Resize ROI to a fixed size (e.g., 64x64)
        resized_roi = cv2.resize(hand_roi, (64, 64))
        
        # Compute Histogram of Oriented Gradients (HOG) features
        hog = cv2.HOGDescriptor()
        features = hog.compute(resized_roi)
        features = features.reshape(1, -1)
        
        # Classify the ROI using SVM classifier
        prediction = clf.predict(features)
        
        # Display the result (e.g., Open/Closed hand)
        if prediction == 1:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, 'Open Hand', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(image, 'Closed Hand', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image

# Main function to process images
def process_images(image_paths):
    for img_path in image_paths:
        # Read the image
        image = cv2.imread(img_path)
        
        # Detect and classify hands
        result_image = detect_and_classify_hands(image)
        
        # Display the result
        cv2.imshow('Hand Recognition', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # List of image paths
    image_paths = glob.glob("/path/to/images/*")
    
    # Process the images
    process_images(image_paths)
