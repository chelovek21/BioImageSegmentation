import glob
import os
import progressbar
import numpy as np

from skimage.morphology import disk
from skimage.filters import median
from scipy import ndimage

from PIL import Image

from image_segmenter.metrics import hausdorff_object_score, object_f_score, dice_object_score

__author__ = "Rasmus Hvingelby"

ground_truth_folder = "/home/hvingelby/Workspace/medical_image_seg/gland"
predictions_folder = "/home/hvingelby/Workspace/medical_image_seg/SpecialCourse_MedicalImageSeg/image_segmenter/my_preds"

hausdorff_scores = []
f1_scores = []
dice_scores = []
bar = progressbar.ProgressBar()
for filename in bar(glob.glob(predictions_folder + '/*.bmp')):
    img = Image.open(filename)
    img_id = int(filename.split('/')[-1].split('.')[0])
    gt_img = Image.open(ground_truth_folder + '/train_'+str(img_id+1)+'_anno.bmp')

    img = img.resize((512, 384), Image.ANTIALIAS)
    img = ndimage.binary_fill_holes(np.array(img))
    img = median(img, disk(3))
    #print(np_img.shape)
    #exit("hey")
    gt_img = gt_img.resize((512, 384), Image.ANTIALIAS)


    hausdorff_scores.append(hausdorff_object_score(np.array(gt_img), np.array(img)))
    f1_scores.append(object_f_score(np.array(gt_img), np.array(img)))
    dice_scores.append(dice_object_score(np.array(gt_img), np.array(img)))

print np.mean(hausdorff_scores)
print np.mean(f1_scores)
print np.mean(dice_scores)