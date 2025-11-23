import cv2

img = cv2.imread("./res/faces.jpg")

cv2.imshow("test-image", img)

cv2.waitKey(0)

cv2.destroyAllWindows()
