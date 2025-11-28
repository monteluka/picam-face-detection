import cv2
import numpy as np
from picamera2 import Picamera2
from ultralytics import YOLO

# starts the picam2 camera object and returns it
def initiateCamera():
    camera = Picamera2()
    # set small resolution so we can process faces faster
    # also set format to BGR from the start so we don't have
    # to convert the image ourselves
    cameraConfig = camera.create_video_configuration(
        main={"size": (640,480),
              "format": "BGR888"}
    )
    camera.configure(cameraConfig)
    camera.start()
    return camera


# number has to be in range [0, 15]
def getCatHead(catHeadsImg, number):
    height = 997
    width = 1000
    y = [0, int(height/4-60), int(height/2-60), int(3*height/4-80), int(height-80)]
    x = [0, int(width/4), int(width/2), int(3*width/4), int(width)]
    i = number // 4
    j = number % 4
    smallerImg = catHeadsImg[y[j+0]:y[j+1],x[i+0]:x[i+1],:]
    return smallerImg

# returns a list where pixels that are background are marked as 0 (0x0)
# and foreground pixels are marked as 255 (0xFF)
def createMask(catHeadImg):
    # first convert the image to grayscale for easier processing
    greyCatHeadImg = cv2.cvtColor(catHeadImg, cv2.COLOR_BGR2GRAY)

    # now set up the thresholds for what colors we consider background
    _, threshold = cv2.threshold(greyCatHeadImg, 240, 255, cv2.THRESH_BINARY_INV)

    # now detect contours (shapes)
    # this will be used to find the shape of the cat's face
    # solves the problem of the white inside cat's face being selected for removal as well
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # the largest contour in the image will be the cat's face
    catHeadContour = max(contours, key=cv2.contourArea)

    # create a mask with the same height and width as the image
    mask = np.zeros_like(greyCatHeadImg)

    # fill the elements in the mask that correspond with the foreground area of the image with white pixels (0xFF)
    cv2.drawContours(mask, [catHeadContour], -1, 255, thickness=cv2.FILLED)

    return mask

def main():
    # start the camera with our needed config
    camera = initiateCamera()

    # get the image that has all the cat heads
    catHeadsImg = cv2.imread("./res/cute-cartoon-kitten-faces.jpg")

    # set up the face detection model
    faceDetector = YOLO("./data/yolo11n.pt")

    # while loop that will get image from camera and manipulate it if necesarry
    while True:
        # get the image from the camera
        img = camera.capture_array()

        # get the info on the face box
        detectedFaceInfo = faceDetector(img, stream=True, verbose=False, conf=0.8)

        # draw over the detected face(s)
        for faces in detectedFaceInfo:
            for face in faces.boxes:
                # set up the face area in a way that cv2 expects
                x, y, x2, y2 = map(int, face.xyxy[0])
                width = x2 - x
                height = y2 - y
                # draw a rectangle over detected face
                cv2.rectangle(img, (x,y), (x+width, y+height), (255,0,0), 5)
                # now place cat head over face
                catHeadImg = getCatHead(catHeadsImg, 0)
                catHeadImg = cv2.resize(catHeadImg, (width, height))
                # create a mask of the image the separates background and foreground
                catHeadMask = createMask(catHeadImg)
                # define the pixels that make up the detected face region
                faceArea = img[y:y+height, x:x+width]
                # pixels in the facial area will be replaced by pixels from the cat head image
                # where pixels in the mask are marked as foreground (0xFF)
                faceArea[catHeadMask == 255] = catHeadImg[catHeadMask == 255]

        # display the final image with detection borders
        cv2.imshow("Detected Heads", img)

        # wait for the window to update
        if cv2.waitKey(1) > -1:
            break

    # free resources
    cv2.destroyAllWindows()
    camera.stop()

if __name__ == "__main__":
    main()
