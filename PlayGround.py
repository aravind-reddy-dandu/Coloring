import PIL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import random


def rgb2gray(rgb):
    # return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])
    return np.dot(rgb[..., :3], [0.21, 0.72, 0.07])


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.classifications = {}
        self.centroids = {}
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        for i in range(self.k):
            rand = random.randint(0, len(data))
            self.centroids[i] = data[rand]

        for i in range(self.max_iter):

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        labels = []
        for pt in data:
            distances = [np.linalg.norm(pt - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            labels.append(classification)
        return labels


path = 'D:\\Study\\Intro_AI\\Coloring_Assignment\\Images\\reduced_300.png'
rgba_image = PIL.Image.open(path)
rgb_image = rgba_image.convert('RGB')
# img = mpimg.imread(path)
# img = img * 255
img = np.array(rgb_image)
gray = rgb2gray(img)
left_actual = img[:, :int(img.shape[1] / 2), :]
left_half = gray[:, :int(img.shape[1] / 2)]
right_half = gray[:, int(img.shape[1] / 2):]
left_actual_flat = left_actual.reshape((-1, 3))

# -------------------
# kmeans = KMeans(n_clusters=5, random_state=0).fit(left_actual_flat)
# labels = kmeans.labels_.reshape(left_actual.shape[0], left_actual.shape[1])
# centers = kmeans.cluster_centers_

m_kmeans = K_Means(k=5, max_iter=1000, tol=0.001)

m_kmeans.fit(left_actual_flat)
labels = np.array(m_kmeans.predict(left_actual_flat)).reshape(left_actual.shape[0], left_actual.shape[1])
centers = m_kmeans.centroids

# ---------------------

colored_left = np.zeros(left_actual.shape)
for i in range(labels.shape[0]):
    for j in range(labels.shape[1]):
        label = labels[i][j]
        cluster_center = centers[label]
        colored_left[i][j] = cluster_center
plt.imshow(colored_left / 255)
plt.show()
# print(kmeans)


# plt.imshow(left_half, cmap=plt.get_cmap('gray'))
# plt.savefig('right_half.png', bbox_inches='tight', pad_inches=0)
# plt.show()
# plt.imshow(right_half, cmap=plt.get_cmap('gray'))
# plt.savefig('left_half.png', bbox_inches='tight', pad_inches=0)
# plt.show()
