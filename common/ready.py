import datetime
import os
from .csver import writer_csv
from .logger import load_logger

__all__ = ['Args']

class Args():
    def __init__(self, dst_dir, model_name):
        # [epoch, loss, acc, miou, mdice,kappa,macro_f1]
        demo_predict_data_headers = ["epoch", "loss", "acc", "miou", "recall", "Kappa", 'Macro_f1']
        self.img_ab_concat = False
        self.en_load_edge = False
        self.num_classes = 0
        self.batch_size = 0
        self.iters = 0

        self.pred_idx = 0
        self.data_name = ""
        time_flag = datetime.datetime.strftime(datetime.datetime.now(), r"%Y_%m_%d_%H")
        self.save_dir = os.path.join(dst_dir, f"{model_name}_{time_flag}")
        self.model_name = model_name

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_predict = os.path.join(self.save_dir , "predict")
        if not os.path.exists(self.save_predict):
            os.makedirs(self.save_predict)

        self.best_model_path = os.path.join(self.save_dir, "{}_best.pdparams".format(model_name))
        log_path = os.path.join(self.save_dir, "train_{}.log".format(model_name))
        self.metric_path = os.path.join(self.save_dir, "{}_metrics.csv".format(model_name))
        print("log save at {}, metric save at {}, weight save at {}".format(log_path, self.metric_path, self.best_model_path))
        self.epoch = 0
        self.loss = 0
        self.logger = load_logger(log_path)
        self.logger.info("log save at {}, metric save at {}, weight save at {}".format(log_path, self.metric_path, self.best_model_path))
        # writer_csv(self.metric_path, headers=demo_predict_data_headers)

