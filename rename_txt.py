from glob import glob
import shutil
import os
from PIL import Image
#0.0004,0.00045,0.0005,0.00055,0.0006
data_path = './results/ADRNet/'
tracker_name = 'ADRNet'
# des_path = '/home/zpy/Desktop/RGBT234_official_evaluation/MPR-MSR-Evaluation/BBresults_ADRNet_upload/'
des_path = '/home/zpy/Desktop/GTOT_official_evaluation/PlotErr50/BBresults_ADRNet_upload/'
if not os.path.exists(des_path):
    os.makedirs(des_path)
data = glob(os.path.join(data_path,'*.txt'))
for x in data:
    txt_name = x.split('/')[-1]
    video_name = txt_name.split('_')[-1]
    if video_name == 'time.txt':
        continue
    else:
        des_name = des_path + tracker_name + '_' + video_name
        shutil.copyfile(x,des_name)