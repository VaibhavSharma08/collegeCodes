import cv2
import numpy as np

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

# import cascade file for facial recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

'''
    # if you want to detect any object for example eyes, use one more layer of classifier as below:
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
'''

while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #
    # ## Generate masks to detect red color
    # lower_red = np.array([0, 120, 50])
    # upper_red = np.array([10, 255, 255])
    # mask1 = cv2.inRange(hsv, lower_red, upper_red)
    #
    # lower_red = np.array([170, 120, 70])
    # upper_red = np.array([180, 255, 255])
    # mask2 = cv2.inRange(hsv, lower_red, upper_red)
    #
    # mask1 = mask1 + mask2
    #
    # ## Open and Dilate the mask image
    # mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    #
    # ## Create an inverted mask to segment out the red color from the frame
    # mask2 = cv2.bitwise_not(mask1)
    #
    # ## Segment the red color part out of the frame using bitwise and with the inverted mask
    # res1 = cv2.bitwise_and(img, img, mask=mask2)
    '''
    # detecting eyes
    eyes = eyeCascade.detectMultiScale(imgGray)
    # drawing bounding box for eyes
    for (ex, ey, ew, eh) in eyes:
        img = cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 3)
    '''

    cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyWindow('face_detect')