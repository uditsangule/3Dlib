from datetime import datetime , timedelta
import yaml
import os

def Fstatus(fpath):
    """
    checks the folder path , if doesn't exist, it creates that path
    """
    if not os.path.exists(fpath):os.makedirs(fpath, exist_ok=True)
    return fpath

def get_currDT(format = "%Y%m%d_%H-%M", timeonly=False):
    """
    gives the current date,time in given format
    """
    if timeonly:
        return datetime.now().strftime(format).split('_')[1]
    return datetime.now().strftime(format)

def read_calibration(calib_path,camid=1):
    file_name = calib_path + os.sep + f'Calibration_Camera{camid}.yaml'
    with open(file_name , 'r') as f:
        calibration = yaml.load(f, Loader=yaml.FullLoader)
    return calibration

def save_calibration(outputdir , data , camid=1):
    outputdir = Fstatus(outputdir)
    with open(outputdir + os.sep + f'Calibration_Camera{camid}.yaml', 'w') as f:
        yaml.dump(data, f, sort_keys=False)
    return
