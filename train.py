from __future__ import division
import json
import os
import shutil

from active_learner import ActiveLearner
from utils import should_evaluate
from model import SegmentModel
from preprocess import GlandHandler
import sys
from metrics import get_scores

import numpy as np


__author__ = "Mathias Baltzersen and Rasmus Hvingelby"

data_path = sys.argv[1]

hps = {"bs": 1,
       "epochs": 1000,  # 400
       "big_k": 2,
       "small_k": 2,  # To run without representativeness set small_k == big_k
       "dropout_prob": 0.5,
       "num_mc_samples": 1, # Number of times we do mc sampling, i.e. pass pool data through network with different dropout
       "acquisition": "variance",  # choose from 'variance', 'entropy', 'KL_divergence' and 'random'
       "exp_name": "my_testing",  # last 3 are big_k, small_k and num_mc_samples
       "initial_training_examples": 5,
       "anno_wait_time": 0.5,  # number of minutes allowed between each oracle query for more data
       "ensemble_method": "dropout",  # choose from 'dropout' and 'bootstrap'
       "lr": 0.0005,
       "img_size": 192,
       "active_learning": True,
       "ensemble_size": 4, #For bootstrap, this determines the number of models trained, otherwise # of ensemble dropout methods
       "classes": 2,
       "scale_nc": 1, # The scaling factor for number of channels. Paper does not scale but their code scales with 2
       "contour_loss_weight": 1.0,
       "l2_scale": 0.0,
       "threshold": 0.5
       }

print("Selected image size......: " + str(hps.get("img_size")))
print("Dataset path.............: " + data_path)

path = "./" + hps.get("exp_name")
paths = [path + "/model_" + str(i) for i in range (4)] if hps.get("ensemble_method")=='bootstrap' else []
paths.insert(0, path)
hps_file = "/hps.json"

for p in paths:
    if not os.path.exists(p):
        os.makedirs(p)
    else:
        shutil.rmtree(p)
        os.makedirs(p)

with open(path + hps_file, 'w') as file:
    json.dump(hps, file)

x_train, y_train_seg, y_train_cont, x_a_test, y_a_test, x_b_test, y_b_test = GlandHandler(data_path).get_gland()

#TODO: remove this, only for fast testing!!
x_train = x_train[:8]
y_train_seg = y_train_seg[:8]
y_train_cont = y_train_cont[:8]

total_num_train_images = x_train.shape[0]

net = SegmentModel(hps)

if not hps.get("active_learning"):
    net.train(x_train, y_train_seg, y_train_cont)
    results_a = net.evaluate(x_a_test, y_a_test)
    results_b = net.evaluate(x_b_test, y_b_test)

    final_pred_a = net.final_predictions(results_a)
    final_pred_b = net.final_predictions(results_b)

    get_scores(final_pred_a, y_a_test, "test_a", hps=hps)
    get_scores(final_pred_b, y_b_test, "test_b", hps=hps)


else:
    al = ActiveLearner(x_train, y_train_seg, y_train_cont, hps)
    x_train, y_train, train_y_cont = al.get_initial_trainingdata()
    pool = al.get_pool()

    # Train network initially
    pool_predictions, pool_image_descriptors = net.train_active(x_train, y_train, train_y_cont, pool)

    # Obtain new data for training
    x_train, y_train, train_y_cont = al.get_training_data(pool_predictions, pool_image_descriptors)

    # Evaluate
    evaluate_at = [0.9, 0.9, 0.9] # Evaluate at these percentage of data used
    percentage_data_in_training = hps.get("initial_training_examples") / total_num_train_images
    evaluate_at, should_eval = should_evaluate(evaluate_at, percentage_data_in_training)

    if should_eval: # TODO: evaluate the net
        pass

    # Active learning loop
    num_learning_loops = int(np.ceil(pool.shape[0] / hps.get("small_k")))
    for i in range(num_learning_loops):
        pool = al.get_pool()
        # Train
        pool_predictions, pool_image_descriptors = net.train_active(x_train, y_train, train_y_cont, pool)

        # Obtain new data for training
        if al.get_pool_size() > 0:
            x_train, y_train, train_y_cont = al.get_training_data(pool_predictions, pool_image_descriptors)

        # Evaluate
        percentage_data_in_training = x_train.shape[0] / total_num_train_images
        evaluate_at, should_eval = should_evaluate(evaluate_at, percentage_data_in_training)

        if should_eval: # TODO: evaluate the net
            pass
