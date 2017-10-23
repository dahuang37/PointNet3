import os
import sys
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import dataset
import data_visualization

data, label = dataset.getTrainingData(0)

index, _ = np.where(label==label[9])
images = data[index]
mean_images = np.mean(images, axis=0)
print(images[1].shape)
print(mean_images.shape)
data_visualization.point_cloud_three_views_demo(images[1],"20")

data_visualization.point_cloud_three_views_demo(mean_images,"20_mean")
