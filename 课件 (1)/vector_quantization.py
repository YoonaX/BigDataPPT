import numpy as np
from scipy import misc
from sklearn import cluster
import matplotlib.pyplot as plt


def compress_image(img, num_clusters):
    X = img.reshape((-1, 1))
    kmeans = cluster.KMeans(n_clusters=num_clusters)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_.squeeze()
    labels = kmeans.labels_
    input_image_compressed = np.choose(labels, centroids).reshape(img.shape)
    return input_image_compressed

if __name__ == '__main__':
    input_file = "iris2.jpg"
    num_clusters = 3
    input_image = misc.imread(input_file, True).astype(np.uint8)
    plt.subplot(121)
    plt.imshow(input_image)
    input_image_compressed = compress_image(input_image, num_clusters)
    plt.subplot(122)
    plt.imshow(input_image_compressed)
    plt.show()
