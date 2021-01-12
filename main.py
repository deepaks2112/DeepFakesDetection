import numpy as np
import os
from preprocessing import Preprocessor
import matplotlib.pyplot as plt
import cv2


fake_path = '/home/deepak/DEEPFAKES_IITBHILAI/dfdc/dfdc_train_part_01/dfdc_train_part_1/bfeewgzrbr.mp4'
real_path = '/home/deepak/DEEPFAKES_IITBHILAI/dfdc/dfdc_train_part_01/dfdc_train_part_1/qjdtgggqym.mp4'

pre = Preprocessor((),32)
masks, reals, fakes = pre.extractDiffMasks(real_path, fake_path)

f, ax = plt.subplots(1, 3, figsize=(15,15))
ax[0].imshow(cv2.cvtColor(reals[3], cv2.COLOR_BGR2RGB))
ax[1].imshow(cv2.cvtColor(fakes[3], cv2.COLOR_BGR2RGB))
ax[2].imshow(masks[3])
plt.show()




