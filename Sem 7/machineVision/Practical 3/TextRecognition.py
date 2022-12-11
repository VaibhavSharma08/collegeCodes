# text recognition
import cv2
import pytesseract
# read image
im = cv2.imread('/Users/vaibhavs/PycharmProjects/collegeCodes/Sem 7/machineVision/Practical 3/hello.png')
cv2.imshow('hello', im)
# configurations
config = ('-l eng --oem 1 --psm 3')
# pytesseract
text = pytesseract.image_to_string(im, config=config)
# print text
text = text.split('\n')
print("The detected and recognized text is -> {}".format(text))

cv2.waitKey(0)
cv2.destroyAllWindows()