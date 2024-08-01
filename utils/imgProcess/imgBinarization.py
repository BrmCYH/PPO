import cv2

def getBinaryImage(image, neighborhood, constant_c):
    if image is None:
        print("Error loading image. Please ensure the file path is correct and the file is an image.")
        return None
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, neighborhood, constant_c)

    return binary_image