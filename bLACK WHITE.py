import cv2

# Load the image using full path
img = cv2.imread('C:/Users/hp/Desktop/py/proj/mee.jpg')

# Check if the image is loaded properly
if img is None:
    print("Image not found or couldn't be loaded.")
    exit()

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Show the original image
cv2.imshow('Original Image', img)

# Show the grayscale image
cv2.imshow('Grayscale Image', gray)

# Wait for a key press and close all OpenCV windows
cv2.waitKey(0)
cv2.destroyAllWindows()

