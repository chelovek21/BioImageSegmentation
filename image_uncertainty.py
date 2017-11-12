import numpy as np
from sklearn import metrics

__author__ = "Mathias Baltzersen"

def uncertainty(image):
    """
    Receives [pool_size, img_h, img_w, softmax_dim] where n is number of dropout experiments, pool_size number of images in pool
    Returns [pool_size, 1]
    """

    max_pixel_prob = 0
    H = image.shape[0]
    W = image.shape[1]
    for h in range(H):
        for w in range(W):
            max_pixel_prob += 1 - np.max(image[h, w, :])
    uncert = (H * W - max_pixel_prob) / (H * W)
    return (uncert)


def representativeness(cosine_sim, k, K, k_treshold):
    K = max(min(K, len(S_c)), k)  # In order to avoid input errors
    k_enum = 1
    b = 0
    S_a = np.array([0])
    S_not_a = np.arange(1, K)
    while (k_enum < k):
        c = k_enum + 1  # Just needs to be bigger than b.
        for n in S_not_a:
            for m in S_a:
                b += cosine_sim[m][n]
            if (b <= c):
                c = b
                nxt_idx = n
            b = 0
        c = c / k_enum  # Not in paper, can be added to yield an inprovement
        if (c >= k_treshold):
            break
        S_a = np.append(S_a, nxt_idx)
        q = np.argwhere(S_not_a == nxt_idx)[0][0]
        S_not_a = np.delete(S_not_a, q)
        k_enum += 1
    return S_a


if __name__ == "__main__":
    # imput is an image consisting of Height, Width and Posterior probabilites.
    image0 = np.random.rand(28, 28, 10)
    image1 = np.random.rand(28, 28, 10)
    image2 = np.random.rand(28, 28, 10)
    image3 = np.random.rand(28, 28, 10)
    image4 = np.random.rand(28, 28, 10)
    image5 = np.random.rand(28, 28, 10)

    image_list = []
    image_list.append(image0)
    image_list.append(image1)
    image_list.append(image2)
    image_list.append(image3)
    image_list.append(image4)
    image_list.append(image5)

    S_u = []

    n = 0
    for image in image_list:
        uncert = uncertainty(image)
        s = [n, uncert]
        n += 1
        S_u.append(s)
    S_u = np.asarray(S_u)
    # S_u =S_u[np.argsort(S_u[:, 1])]
    print(S_u)

    # Returns the index of the images from S_u that should be in S_c
    K = 5
    S_u = S_u.flatten()[::2]
    S_c_index = S_u[0:K]
    print(S_c_index)
    # Now we need to fetch the I_c vectors from the
    # images with the index given by S_c_index.

    uncert_0 = [1, 2, 1, 3, 2, 4, 53]
    uncert_1 = [2, 1, 1, 5, 2, 5, 22]
    uncert_2 = [9, 0, 3, 1, 2, 6, 25]
    uncert_3 = [1, 1, 1, 2, 6, 3, 2]
    uncert_4 = [1, 1, 2, 6, 2, 7, 1]
    uncert_5 = [1, 13, 7, 2, 7, 3, 21]

    S_c = []
    S_c.append(uncert_0)
    S_c.append(uncert_1)
    S_c.append(uncert_2)
    S_c.append(uncert_3)
    S_c.append(uncert_4)
    S_c.append(uncert_5)

    S_not_a = np.arange(1, len(S_c))
    S_a = np.array([0])

    cosine_sim = metrics.pairwise.cosine_similarity(S_c)

    s_a = representativeness(cosine_sim=cosine_sim, k=6, K=6, k_treshold=0.6)

    print(cosine_sim)
    print(s_a)
