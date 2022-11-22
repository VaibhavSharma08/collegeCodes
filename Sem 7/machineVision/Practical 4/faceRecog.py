import cv2
import face_recognition
import time

# ## Preparation for writing the ouput video
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('/Users/vaibhavs/PycharmProjects/machineVision/Practical 4/output.avi', fourcc, 20.0, (640, 480))
#
# imgmain = face_recognition.load_image_file('testFace.jpg')
# imgmain = cv2.cvtColor(imgmain, cv2.COLOR_BGR2RGB)
#
# ##reading from the webcam
# cap = cv2.VideoCapture(0)
#
# ## Allow the system to sleep for 3 seconds before the webcam starts
# time.sleep(3)
# count = 0
# background = 0
#
# ## Read every frame from the webcam, until the camera is open
# while (cap.isOpened()):
#     ret, img = cap.read()
#     if not ret:
#         break
#
#     imgTest = face_recognition.load_image_file(img)
#     imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)
#
#     faceLoc = face_recognition.face_locations(imgmain)[0]
#     encodeElon = face_recognition.face_encodings(imgmain)[0]
#     cv2.rectangle(imgmain, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
#
#     faceLocTest = face_recognition.face_locations(imgTest)[0]
#     encodeTest = face_recognition.face_encodings(imgTest)[0]
#     cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)
#
#     results = face_recognition.compare_faces([encodeElon], encodeTest)
#     faceDis = face_recognition.face_distance([encodeElon], encodeTest)
#     print(results, faceDis)
#     cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#
#     cv2.imshow('Main Image', imgmain)
#     cv2.imshow('Test Image', imgTest)
#     cv2.waitKey(1)
#
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()





imgmain = face_recognition.load_image_file('bryanMain.jpg')
imgmain = cv2.cvtColor(imgmain, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file('bryanTest.jpg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgmain)[0]
encodeElon = face_recognition.face_encodings(imgmain)[0]
cv2.rectangle(imgmain, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results, faceDis)
cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow('Main Image', imgmain)
cv2.imshow('Test Image', imgTest)
cv2.waitKey(0)