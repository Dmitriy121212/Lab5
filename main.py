import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

image = cv2.imread('image1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

k = 3
kmeans = KMeans(n_clusters=k, random_state=0)
labels = kmeans.fit_predict(pixel_values)
centers = np.uint8(kmeans.cluster_centers_)
segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(image.shape)

plt.imshow(segmented_image)
plt.axis('off')
plt.show()
