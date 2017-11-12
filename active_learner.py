import numpy as np
from PIL import Image
from utils import acquisition_func, max_representativeness

__author__ = "Mathias Baltzersen and Rasmus Hvingelby"

class ActiveLearner:

    def __init__(self, x_train, y_train, y_train_cont, hps):
        self.num_init_training_examples = hps.get("initial_training_examples")
        self.num_examples_per_learning_loop = hps.get("small_k") # TODO: other name, this is aka small k
        self.num_most_uncertain_examples = hps.get("big_k")
        self.acquisition_function = hps.get("acquisition")
        self.hps = hps

        # We take images from the original_size_pool and add them to the train_set.
        # When we do the training of the model we chop and augment the images
        # as defined in the augment_batch method from utils.py
        self.original_size_pool = (x_train, y_train, y_train_cont)

        # We use the images from the resized_pool to estimate uncertainty
        # and getting image descriptors. Actually this does not need to
        # have the y_train images.
        self.new_size = 512, 384  # suitable number for resizing regarding memory and proper dimension forward passing
        self.resized_pool = self._resize_images_for_pool(x_train, y_train)

        self.train_set = None

    def get_pool_size(self):
        """
        :return: the number of elements in the pool
        """
        return self.resized_pool[0].shape[0]

    def get_pool(self):
        """
        :return: unlabelled data (x) from the resized_pool
        """
        return self.resized_pool[0]

    def get_initial_trainingdata(self):
        """
        Selects an initial training data set randomly from pool.

        :return: train_x, train_y
        """
        pool_size = self.resized_pool[0].shape[0]

        idx = np.random.choice(pool_size, replace=False, size=self.num_init_training_examples)
        train_x, train_y, train_y_cont = self._update_sets(idx)

        return train_x, train_y, train_y_cont

    def get_training_data(self, predictions, image_descriptors):
        """
        This model will find the best data points from the pool.
        These are calculated based on the acquisition function
        and the representativeness.

        :param image_descriptors: Image descriptor for each image in pool. shape()
        :param predictions: Each model in ensemble's predictions on the pool. shape(num_committee_models, bs, width, height)
        :return: x_train, y_train
        """
        print("Finding new data from pool")
        assert predictions.shape == (self.hps.get("num_mc_samples"), self.resized_pool[0].shape[0], self.new_size[1], self.new_size[0], self.hps.get("classes"))
        assert image_descriptors.shape == (self.resized_pool[0].shape[0], 1024*self.hps.get("scale_nc"))

        if self.acquisition_function == 'random':
            pool_size = self.resized_pool[0].shape[0]
            idx = np.random.choice(pool_size, replace=False, size=self.num_examples_per_learning_loop)
            return self._update_sets(idx)
        else:
            #dropout_predictions = np.array([self.model.get_dropout_predictions(self.pool[0]) for _ in range(self.num_dropout_committee_models)]) # TODO: check shape
            unsorted_best_data = acquisition_func(self.acquisition_function, predictions)

            # Now we sort them and we add the top K. Also remove from pool
            idx = np.argsort(unsorted_best_data)[-self.num_most_uncertain_examples:]

            # Check if we need to do representativeness computation
            if self.num_examples_per_learning_loop < self.num_most_uncertain_examples:
                #image_descriptors = self.model.get_image_descriptors(self.pool[0]) # TODO: Check shape!!

                candidate_set = image_descriptors[idx, :]

                # Representativeness function expects lists
                image_descriptors = np.split(image_descriptors, image_descriptors.shape[0], axis=0)
                candidate_set = np.split(candidate_set, candidate_set.shape[0], axis=0)

                S_a_idcs = max_representativeness(image_descriptors, candidate_set, idx.tolist(), self.num_examples_per_learning_loop)
            else:
                S_a_idcs = idx

        return self._update_sets(S_a_idcs)

    def _update_sets(self, idx):
        """
        Bookkeeping method that takes a set of indexes of data_points and
        removes these from the pool (resized_pool and original_size_pool)
        and add them to the train set.

        :param idx:
        :return train_x, train_y:
        """
        if self.train_set is None:
            train_x = self.original_size_pool[0][idx]
            train_y = self.original_size_pool[1][idx]
            train_y_cont = self.original_size_pool[2][idx]
        else:
            train_x = np.append(self.train_set[0], self.original_size_pool[0][idx], axis=0)
            train_y = np.append(self.train_set[1], self.original_size_pool[1][idx], axis=0)
            train_y_cont = np.append(self.train_set[2], self.original_size_pool[2][idx], axis=0)
        self.train_set = (train_x, train_y, train_y_cont)

        # Remove the pooled samples from resized_pool
        pool_samples = np.delete(self.resized_pool[0], idx, axis=0)
        pool_labels = np.delete(self.resized_pool[1], idx, axis=0)
        self.resized_pool = (pool_samples, pool_labels)

        # Remove the pooled samples from original_size_pool
        original_samples = np.delete(self.original_size_pool[0], idx, axis=0)
        original_labels = np.delete(self.original_size_pool[1], idx, axis=0)
        original_contour_labels = np.delete(self.original_size_pool[2], idx, axis=0)

        self.original_size_pool = (original_samples, original_labels, original_contour_labels)

        return train_x, train_y, train_y_cont

    def _resize_images_for_pool(self, x_train, y_train):
        """
        This method resizes the images in the pool to an image size
        that fits into the network.

        :param x_train:
        :param y_train:
        :return: pool_x_train, pool_y_train
        """
        pool_x_train = []
        pool_y_train = []


        for x_img, y_img in zip(x_train, y_train):
            tmp_y_img = np.argmax(y_img, axis=2)

            tmp_x_img = Image.fromarray(x_img).resize(self.new_size, Image.ANTIALIAS)
            tmp_y_img = Image.fromarray(tmp_y_img, 'L').resize(self.new_size, Image.ANTIALIAS)

            pool_x_train.append(np.array(tmp_x_img))
            pool_y_train.append(np.array(tmp_y_img))

        return np.array(pool_x_train), np.array(pool_y_train)
