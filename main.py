import cv2

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

def main():
    img = cv2.imread("./res/faces.jpg")
    catHeadsImg = cv2.imread("./res/cute-cartoon-kitten-faces.jpg")

    # set up the face cascade
    faceDetector = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')

    # convert image to grayscale
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # get the info on the face box
    detectedFaceInfo = faceDetector.detectMultiScale(imgGray, 1.3, 5)

    # draw over the detected face(s)
    for (x, y, width, height) in detectedFaceInfo:
        # first draw a rectangle over detected face
        cv2.rectangle(img, (x,y), (x+width, y+height), (255,0,0), 5)
        # now place cat head over face
        catHeadImg = getCatHead(catHeadsImg, 0)
        catHeadImg = cv2.resize(catHeadImg, (width, height))
        img[y:y+height, x:x+width] = catHeadImg

    # display the final image with detection borders
    cv2.imshow("Detected Heads", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
