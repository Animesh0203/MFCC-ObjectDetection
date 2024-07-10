import cv2
import numpy as np

# Read image.
image = cv2.imread(r"C:\Users\Noct\Downloads\coins.jpeg", cv2.IMREAD_COLOR)

# Convert to grayscale.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Create a blob detector
params = cv2.SimpleBlobDetector_Params()

# Set the parameters for blob detection
params.minThreshold = 10
params.maxThreshold = 200
params.filterByArea = True
params.minArea = 100
params.filterByCircularity = True
params.minCircularity = 0.8
params.filterByConvexity = False
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the grayscale image
keypoints = detector.detect(blurred)

# Draw detected blobs on the original image
image_with_blobs = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 255, 0),
                                     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display the image with detected blobs
cv2.imshow("Detected Coins", image_with_blobs)
cv2.waitKey(0)

# Print the number of coins detected
print("Number of coins detected:", len(keypoints))

# Clean up
cv2.destroyAllWindows()