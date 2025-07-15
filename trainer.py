import random
import os
import numpy as np
import paddle
import logging

from datasets.dataloader import DataReader, TestReader
from paddleseg.models import UNet, DeepLabV3P, UNetPlusPlus, UPerNet, SegNeXt, ResNet50_vd, Xception65_deeplab
from work.train import train
from sfesam import SFESAM
from common import Args


# 参数、优化器及损失
batch_size = 4
iters = 100
base_lr = 5e-5

dataset_name = "Landcover"
dataset_path = r'/home/aistudio/data/{}'.format(data_name)


num_classes = 5

# model = UNetPlusPlus(num_classes, 3)
# model = UPerNet(num_classes, ResNet50_vd(in_channels=3),(0,1,2,3))
# model = DeepLabV3P(num_classes, backbone=ResNet50_vd(in_channels=6))
# model = SegNeXt(num_classes=num_classes, decoder_cfg={}, backbone=ResNet50_vd(in_channels=6))
# model = DeepLabV3P(num_classes=num_classes, backbone=Xception65_deeplab(), backbone_indices=(0,1))
model = SFESAM(num_classes)

model_name = model.__str__().split("(")[0]
args = Args('output/{}'.format(dataset_name.lower()), model_name)
args.batch_size = batch_size
args.num_classes = num_classes
args.pred_idx = 0
args.data_name = dataset_name
args.device = "gpu:0"
args.img_ab_concat = True
args.en_load_edge = False

def seed_init(seed=32767):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    

if __name__ == "__main__":
    print("main")
    seed_init(32767)
    logging.getLogger('PIL').setLevel(logging.WARNING) # 设置PIL模块的日志等级为WARNING

    train_data = DataReader(dataset_path, 'train', args.en_load_edge, args.img_ab_concat)
    val_data = DataReader(dataset_path, 'val', args.en_load_edge, args.img_ab_concat)
    test_data = TestReader(dataset_path, 'val', args.en_load_edge, args.img_ab_concat)

    lr = paddle.optimizer.lr.CosineAnnealingDecay(base_lr, T_max=(iters // 3), last_epoch=0.5)  
    optimizer = paddle.optimizer.Adam(lr, parameters=model.parameters(),) 
   
    train(model,train_data, val_data, test_data, optimizer, args, iters, 2)

   