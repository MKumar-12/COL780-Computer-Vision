Image Alignment:
    Determine feature points in each image. This could be done using techniques like the Harris corner detector or SIFT.
    Match feature points between adjacent images to find corresponding points.
    Use the matched points to estimate the transformation (e.g., rotation, translation) needed to align the images.

SIFT:
    A]  Constructing a scale space  -   DoG (difference of Guassian)
    B]  Keypoint localization
    C]  Orientation assignment
    D]  Keypoint descriptor

    Constructing a scale space
    1. Get guassian blur of image   {Noise Reduction}
    2. Create 4 scaled versions(octaves) of same image n get their corresponding blurs(x5, using Sigma)
    3. Perform DoG {zyaada blurred img - kam blurred img of org.}
        Now a new set of images will be obtained that'll be used for identifying keypoints.
    
    Keypoint localization
    1. Find local maxima n minima from images
    2. remove low contrast keypoints {keypoints selection}
        Compute 2nd order taylor expansion, if val < 0.03 discard the low contrast keypoints.



Compute transformation matrix btw overlapping images 
Feature matching 
Image Alignment {homography}
Image Stitching {combine@common plane + blending}

===========================================================================================================================================
Image Stitching:
    Once the images are aligned, create a composite image (panorama) by blending overlapping regions.
    Implement a blending algorithm (e.g., linear blending, gradient-based blending) to seamlessly merge the aligned images.


Image Warping (Optional):
    Apply perspective transformation to warp the aligned images onto a common coordinate system.
    This step might be necessary if there are significant perspective distortions between images.


Exposure Compensation (Optional):
    Adjust the brightness and contrast of the images to achieve a consistent exposure across the panorama.
    Techniques such as histogram matching or exposure fusion can be used for this purpose.


Seam Removal (Optional):
    Remove visible seams or artifacts introduced during the blending process.
    Techniques like multi-band blending or graph-cut optimization can be used to minimize visible seams.


Color Correction (Optional):
    Ensure consistent color balance and tone mapping across the panorama.
    Techniques such as histogram equalization or color adjustment can be applied to achieve uniform color appearance.


Final Touch-ups:
    Perform any final adjustments or enhancements to the panorama image.
    This could include cropping the image to remove empty areas or refining the boundaries.


Output:
    Save the resulting panorama image to a file format


================================================================================================================================================
python main.py 1 .\data\Images\Field Res
python main.py 2 .\data\Videos\vid1.mp4 Res