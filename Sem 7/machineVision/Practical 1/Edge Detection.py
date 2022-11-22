import cv2
import matplotlib.pyplot as plt

# Open the image
img = cv2.imread('/Users/vaibhavs/PycharmProjects/machineVision/Practical 1/test1.jpg')

# Apply Canny
edges = cv2.Canny(img, 100, 200, 3, L2gradient=True)
plt.figure()
plt.imsave('/Users/vaibhavs/PycharmProjects/machineVision/Practical 1/edgeDetectedPhoto.png', edges, cmap='gray', format='png')
plt.title('Edge Detected Photo')
plt.imshow(edges, cmap='gray')
plt.show()
