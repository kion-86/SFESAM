import cv2
import numpy as np
import paddle
import paddle.nn.functional as F

from datasets.dataloader import TestReader
from work.predict import predict
from f3net import F3Net




dataset_name = "Landcover"

dataset_path = '/home/aistudio/data/{}'.format(dataset_name)
num_classes = 2

model = F3Net(num_classes = num_classes)

datatest = TestReader(dataset_path)

if __name__ == "__main__":
    print("test")
    weight_path = r"/home/aistudio/data/data250593/f3net_gvlm_cd.pdparams"
    predict(model, datatest, weight_path, datatest.data_name, num_classes, "./output")