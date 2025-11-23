import cv2

img = cv2.imread("./res/faces.jpg")

# set up the face cascade
faceDetector = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

# convert image to grayscale
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# get the info on the face box
detectedFaceInfo = faceDetector.detectMultiScale(imgGray, 1.3, 5)

# draw a rectangle over the detected face(s)
for (x, y, width, height) in detectedFaceInfo:
    cv2.rectangle(img, (x,y), (x+width, y+height), (255,0,0), 5)

# display the final image with detection borders
cv2.imshow("Detected Heads", img)
cv2.waitKey(0)
cv2.destroyAllWindows()