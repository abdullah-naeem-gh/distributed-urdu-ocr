import cv2
import numpy as np

def segment_lines(image_path):
    """
    Reads an image from disk and segments it into individual lines 
    using a horizontal projection profile.
    
    Returns:
        List of cropped line images (numpy arrays).
    """
    # Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
        
    # Binarize image (invert so text is white, background is black for projection)
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Calculate horizontal projection profile
    # Sum of pixel values along each row
    proj = np.sum(thresh, axis=1)
    
    # Find peaks and valleys
    # A valley (0 or very low sum) indicates space between lines
    threshold_val = np.max(proj) * 0.05 # 5% of max peak to filter noise
    
    line_regions = []
    in_line = False
    start_y = 0
    
    for y, val in enumerate(proj):
        if val > threshold_val and not in_line:
            in_line = True
            start_y = max(0, y - 5) # add small padding
        elif val <= threshold_val and in_line:
            in_line = False
            end_y = min(img.shape[0], y + 5)
            # Only consider it a line if it's tall enough (filter out specks)
            if end_y - start_y > 15:
                line_regions.append((start_y, end_y))
                
    # If we reached the end of the image while in a line
    if in_line:
        end_y = img.shape[0]
        if end_y - start_y > 15:
            line_regions.append((start_y, end_y))
            
    # Crop the original grayscale image using the found regions
    line_images = []
    for (y1, y2) in line_regions:
        line_images.append(img[y1:y2, :])
        
    # If segmentation failed completely, return the whole image as one line
    if not line_images:
        return [img]
        
    return line_images
