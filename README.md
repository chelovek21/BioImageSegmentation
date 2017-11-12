# BioImageSegmentation
Implementation of [Suggestive Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation]( https://arxiv.org/abs/1706.04737 "paper")
This includes multiple uncertainty functions for active learning loop and implementation for dropout to estimate uncertainty instead of boostrap method described in the paper.

The architecture used is from [DCAN: Deep Contour-Aware Networks for Accurate Gland Segmentation](https://arxiv.org/abs/1604.02677 "paper")

## Dataset
The dataset used is the gland dataset. Its is available at:
https://www2.warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/download/warwick_qu_dataset_released_2016_07_08.zip.
Download and unzip. Point to the folder in the run script(run.sh).

## Details
TODO: description of each file
Insert test predictions

## Status
Active learning part might not be working!

I suspect that the some of the metrics are not calculated right, since the score does not really reflect the image outputted by the network.

## TODO
- [ ] Make sure active learner is working
- [ ] Fix metrics
- [ ] Implement NasNet cells in the network
