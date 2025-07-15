import sys
import csv
import logging
from logging.handlers import RotatingFileHandler


#show or save figure
#mode=["plt.save","cv2.write"] = [0,1]
def load_logger(save_log_dir,save=True,print=True,config=None):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if print:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if save:
        file_handler = RotatingFileHandler(save_log_dir, maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if config != None:
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger


if __name__ == "__main__":
    print("data_reader.utils run")
    x= ['1','3','5','7','9','0','2','4','6','8']
    with open("../snapshot/levir/metrics.csv","w",newline="") as csv_file:
        filewrite = csv.writer(csv_file)
        filewrite.writerow(["epoch","loss","PA","PA_Class","mIoU","FWIoU","Kappa",'Macro_f1'])
