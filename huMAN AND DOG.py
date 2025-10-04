import cv2
import sys

# Print OpenCV version
print("OpenCV version:", cv2.__version__)

# Load the pre-trained Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Full path to the image file (make sure the path is correct)
image_path = 'C:/Users/hp/Desktop/py/proj/yyy.jpg'

# Load the image
image = cv2.imread(image_path)

# Check if image loaded successfully
if image is None:
    print(f"Error: Could not load image at {image_path}")
    sys.exit()

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

# Print number of faces detected
print(f"Number of faces detected: {len(faces)}")

# Draw rectangles around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Display the image with detected faces
cv2.imshow("Detected Faces", image)

# Save the result to a file
cv2.imwrite("detected_faces.jpg", image)

# Wait until a key is pressed, then close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
