import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing
from torch.utils.data.dataloader import DataLoader
import cv2

sys.path.append('..')
from model.my_network import FingerGAN
from util.my_data_loader import LatentDataset
from util.utils import get_gpus_memory_info, calc_parameters_count, get_logger

torch.multiprocessing.set_sharing_strategy('file_system')


class TestNetwork(object):

    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_model()
        self._check_resume()

    def _init_configure(self):
        parser = argparse.ArgumentParser(description='Process the command line.')
        parser.add_argument('--latent_fingerprint_dir', type=str, default=r'..\Dataset\SD27_latent_TV_texture')
        parser.add_argument('--out_dir', type=str, default='..\enhancement_results')
        parser.add_argument('--resume_path', type=str, default=r'..\trained_model\FingerGAN_25850831.tar')
        parser.add_argument('--step', type=int, default=10) # the step for sliding windows. The smaller, the better the enhancement accuracy, but the more the running time
        parser.add_argument('--en_batch_size', type=int, default=48) # batch size to enhance a latent. A large value can speed up the computation. 
        parser.add_argument('--gpu_num', type=int, default=1)
        parser.add_argument('--num_works', type=int, default=4)
        parser.add_argument('--file_num', type=int, default=None)
        parser.add_argument('--file_start', type=int, default=None)

        args = parser.parse_args()
        self.args = args

    def _init_logger(self):

        self.args.out_dir = self.args.latent_fingerprint_dir+'_enhancement' if self.args.out_dir is None else self.args.out_dir
        if self.args.out_dir is not None and not os.path.exists(self.args.out_dir):
            os.makedirs(self.args.out_dir)

    def _init_device(self):
        if self.args.gpu_num == -1:
            print('no gpu device available')
            cudnn.enabled = False
            cudnn.benchmark = False
            self.multi_GPU = False
            self.device = torch.device('cpu')
        else:
            if not torch.cuda.is_available():
                print('no gpu device available')
                sys.exit(1)
            cudnn.enabled = True
            cudnn.benchmark = True
            self.device_id, self.gpus_info = get_gpus_memory_info()
            self.multi_GPU = True if self.args.gpu_num > 1 else False
            self.device = torch.device('cuda:{}'.format(0 if self.multi_GPU else self.device_id))

    def _init_model(self):
        self.input_row = 192
        self.input_col = 192
        self.FingerGAN_model = FingerGAN(in_channels=1).to(self.device)

    def _check_resume(self):

        resume_path = self.args.resume_path if self.args.resume_path is not None else None
        if resume_path is not None:
            if os.path.isfile(resume_path):
                resume_model = torch.load(resume_path, map_location=self.device)
                self.FingerGAN_model.load_state_dict(resume_model['UNet_model_state'])
            else:
                print("No resume_model found at '{}'".format(resume_path))
                sys.exit(1)
                
        if self.args.gpu_num > -1:
            if torch.cuda.device_count() > 1 and self.multi_GPU:
                print('use: {} gpus'.format(torch.cuda.device_count()))
                self.FingerGAN_model = nn.DataParallel(self.FingerGAN_model)
            else:
                print('gpu device = {}'.format(self.device_id))
                torch.cuda.set_device(self.device_id)

            print('FingerGAN param size = {}MB'.format(calc_parameters_count(self.FingerGAN_model)))

    def run_test(self):
        self.FingerGAN_model.eval()

        with torch.no_grad():
            file_list = os.listdir(self.args.latent_fingerprint_dir)
            
            file_num = self.args.file_num if self.args.file_num is not None else len(file_list)
            file_start = self.args.file_start if self.args.file_start is not None else 0
            
            for ind in range(file_start,file_start+file_num):
                file_name = file_list[ind]

                file_path = os.path.join(self.args.latent_fingerprint_dir, file_name)
                img = cv2.imread(file_path, 0)
                img = img.astype(np.float32) / 255.0
                
                img = torch.from_numpy(img)

                en_latent = self.latentEnhance(img)
                if len(en_latent) == 0:
                    continue
                en_latent = en_latent.cpu()
                #xmin = en_latent.min()
                #xmax = en_latent.max()
                #en_latent = (en_latent-xmin)/(xmax-xmin)
                out_img = en_latent.numpy()
    
                out_file_path = os.path.join(self.args.out_dir, file_name)
                cv2.imwrite(out_file_path, out_img*255)


    def latentEnhance(self, latent):
        shape_latent = latent.shape
        ROW = shape_latent[0]
        COL = shape_latent[1]

        if ROW < self.input_row | COL < self.input_col:
            print("The Row and Col must >= 192")
            _latent_en = []
        else:
            dataset_test = LatentDataset(latent, self.input_row, self.input_col, self.args.step)
            patch_generator_test = DataLoader(dataset_test, batch_size=self.args.en_batch_size, shuffle=False, num_workers=self.args.num_works, drop_last=False)
            latent_en = torch.zeros(ROW, COL).to(self.device)
            mask = torch.zeros(ROW, COL).to(self.device)

            for step, (patch_ind, patch) in enumerate(patch_generator_test):
                
                patch = patch.to(self.device)
                
                en_patch = self.FingerGAN_model(patch)

                for ind in range(0,patch_ind.shape[0]):
                    row_ind = patch_ind[ind,0]
                    col_ind = patch_ind[ind,1]

                    latent_en[(row_ind-self.input_row):row_ind, (col_ind-self.input_col):col_ind] += en_patch[ind,0]
                    mask[(row_ind-self.input_row):row_ind, (col_ind-self.input_col):col_ind] += 1

            mask[mask==0] = 1
            _latent_en = latent_en / mask
        return _latent_en


if __name__ == '__main__':
    train_network = TestNetwork()
    train_network.run_test()
