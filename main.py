import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import dataset
import data_visualization

data, label = dataset.getTrainingData(0)
OFF_FILE = "data/ModelNet40/airplane/test/airplane_0627.off"
data_visualization.draw_mesh(OFF_FILE)
print(label[0])

