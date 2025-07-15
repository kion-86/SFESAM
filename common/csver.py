import cv2
import csv
import pandas as pd
import numpy as np
import math

#create and get label information
def writer_csv(csv_dir,operator="w",headers=None,lists=None):
    with open(csv_dir,operator,newline="") as csv_file:
        f_csv=csv.writer(csv_file)
        if headers!=None:
            f_csv.writerow(headers)
        if lists!=None:
            f_csv.writerows(lists)

def save_numpy_as_csv(scv_dir,d_numpy,fmt="%.4f"):
    assert len(d_numpy.shape) <= 2
    if len(d_numpy.shape)==1:
        d_numpy = np.expand_dims(d_numpy, 0)
    with open(scv_dir,"a") as f:
        np.savetxt(f, d_numpy, fmt=fmt,delimiter=',')

def reader_csv(csv_dir):
    ann = pd.read_csv(csv_dir)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label

def read_excel(path):
    dara_xls = pd.ExcelFile(path)
    data = {}
    for sheet in dara_xls.sheet_names:
        df = dara_xls.parse(sheet_name=sheet,header=None)
        #print(type(df.values))
        data[sheet] = df.values
    return data

def read_csv(csv_dir):
    data = pd.read_csv(csv_dir).values
    return data

def reverse_one_hot(image):
    image = np.array(image.cpu())
    image = np.transpose(image, (1, 2, 0))
    x = np.argmax(image, dim=-1)
    return x

def colour_code_segmentation(image, label_values):
	"""
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

	# w = image.shape[0]
	# h = image.shape[1]
	# x = np.zeros([w,h,3])
	# colour_codes = label_values
	# for i in range(0, w):
	#     for j in range(0, h):
	#         x[i, j, :] = colour_codes[int(image[i, j])]
	label_values = [label_values[key] for key in label_values]
	colour_codes = np.array(label_values)
	image = np.array(image.cpu())
	x = colour_codes[image.astype(int)]
	return x

def scale_image(input,factor):
    #效果不理想，边缘会有损失，不建议使用 2020/5/17 hjq
    #input.shape=[m,n],output.shape=[m//factor,n//factor]
    #将原tensor压缩factor

    h=input.shape[0]//factor
    w=input.shape[1]//factor

    return cv2.resize(input,(w,h),interpolation=cv2.INTER_NEAREST)

class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10



if __name__ == "__main__":
    print("data_reader.utils run")
    x= ['1','3','5','7','9','0','2','4','6','8']
    with open("../snapshot/levir/metrics.csv","w",newline="") as csv_file:
        filewrite = csv.writer(csv_file)
        filewrite.writerow(["epoch","loss","PA","PA_Class","mIoU","FWIoU","Kappa",'Macro_f1'])
