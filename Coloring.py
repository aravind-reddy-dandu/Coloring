import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import PIL.Image
from skimage.util import view_as_blocks
from sklearn.feature_extraction import image
from sklearn.neighbors import NearestNeighbors
import numpy_indexed as npi


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.21, 0.72, 0.07])


def extract_blocks(a, blocksize, keep_as_view=False):
    M, N = a.shape
    b0, b1 = blocksize
    if keep_as_view == 0:
        return a.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2).reshape(-1, b0, b1)
    else:
        return a.reshape(M // b0, b0, N // b1, b1).swapaxes(1, 2)


class Coloring:
    def __init__(self, img_path):
        rgba_image = PIL.Image.open(img_path)
        rgb_image = rgba_image.convert('RGB')
        self.img = np.array(rgb_image)
        self.img = self.img
        gray = rgb2gray(self.img)
        self.left_actual = self.img[:, :int(self.img.shape[1] / 2), :]
        self.left_half = gray[:, :int(self.img.shape[1] / 2)]
        self.right_half = gray[:, int(self.img.shape[1] / 2):]

    def k_means(self, img_part, num_clusters):
        flattened = img_part.reshape((-1, 3))
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(flattened)
        labels = kmeans.labels_.reshape(img_part.shape[0], img_part.shape[1])
        return labels, kmeans.cluster_centers_

    def naive_paint_left_half(self, num_clusters):
        labels, centers = self.k_means(self.left_actual, num_clusters)
        colored_left = np.zeros(self.left_actual.shape)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                label = labels[i][j]
                cluster_center = centers[label]
                colored_left[i][j] = cluster_center
        return colored_left

    def extract_patches_get_knn(self, k):
        pass


coloring = Coloring('D:\\Study\\Intro_AI\\Coloring_Assignment\\Images\\reduced.png')
# left_patches = extract_blocks(coloring.left_half, (3, 3))
patches = image.extract_patches_2d(coloring.left_half, (3, 3))
patches = patches.reshape((-1, 9))
nbrs = NearestNeighbors(n_neighbors=6).fit(patches)
nbrs.kneighbors(patches[1], 2, return_distance=False)
x = coloring.left_half
colored_left = coloring.naive_paint_left_half(5)
plt.imshow(colored_left / 255)
plt.show()
