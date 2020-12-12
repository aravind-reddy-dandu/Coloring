import numpy as np
import random


# K_means class
class K_Means_Clustering:
    # Initializing number of clusters, threshold and max iterations
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.clusters = {}
        self.centers = {}
        self.k = k
        self.tol = tol
        self.maximum_iters = max_iter

    # Fitting over data
    def fit(self, image_part):

        # Creating random centers for the first time
        for i in range(self.k):
            rand = random.randint(0, len(image_part))
            self.centers[i] = image_part[rand]

        # Looping over n iters
        for i in range(self.maximum_iters):

            for i in range(self.k):
                self.clusters[i] = []

            # looping over features
            for columns in image_part:
                # Using euclidean distance
                distances = [np.linalg.norm(columns - self.centers[centroid]) for centroid in self.centers]
                # Updating clusters
                labels = distances.index(min(distances))
                self.clusters[labels].append(columns)

            prev_centers = dict(self.centers)

            # Updating centers
            for labels in self.clusters:
                self.centers[labels] = np.average(self.clusters[labels], axis=0)

            optimized = True

            # Check if threshold is reached
            for c in self.centers:
                org_centers = prev_centers[c]
                curr_centers = self.centers[c]
                if np.sum((curr_centers - org_centers) / org_centers * 100.0) > self.tol:
                    # print(np.sum((curr_centers - org_centers) / org_centers * 100.0))
                    optimized = False

            # Break if threshold is reached
            if optimized:
                break

    # Function to find labels for any data using the fitted data
    def find_labels(self, data):
        labels = []
        for pt in data:
            distances = [np.linalg.norm(pt - self.centers[centroid]) for centroid in self.centers]
            classification = distances.index(min(distances))
            labels.append(classification)
        return labels
