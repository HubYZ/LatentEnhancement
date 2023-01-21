# FingerGAN: A Constrained Fingerprint Generation Scheme for Latent Fingerprint Enhancement

Latent fingerprint enhancement implemented using Pytorch=1.10, and test in Linux 4.18 with a GPU

## Step 1
Install a pytorch environment according to the requirements.txt

## Step 2
Generate latent fingerprint textures using 'Cartoon-Texture Image Decomposition'
	-- code for 'Cartoon-Texture Image Decomposition': http://www.ipol.im/pub/art/2011/blmv_ct/
	-- the parameter sigma in this code was set to 6
	
## Step 3 
Latent enhancement using the FingerGAN
	-- download the trained model https://drive.google.com/file/d/1lS8_wbMjf0_AHYMG4mO_Mrb2TpFVQVTa/view?usp=share_link and place it in the folder ./trained_model 
	-- according to your image location, adjust the related parameters in the function '_init_configure()' in the file ./experiment/latent_fingerprint_enhancement.py
	-- run latent_fingerprint_enhancement.py to obtain the enhanced fingerprints
	

## Citations

Please cite the following papers:

1. Zhu, Yanming, Xuefei Yin, and Jiankun Hu. "FingerGAN: A Constrained Fingerprint Generation Scheme for Latent Fingerprint Enhancement." IEEE Transactions on Pattern Analysis and Machine Intelligence (2023).
```
@article{zhu2023fingergan,
  title={FingerGAN: A Constrained Fingerprint Generation Scheme for Latent Fingerprint Enhancement},
  author={Zhu, Yanming and Yin, Xuefei and Hu, Jiankun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
}
```
2. Zhu, Yanming, Xuefei Yin, Xiuping Jia, and Jiankun Hu. "Latent fingerprint segmentation based on convolutional neural networks." In 2017 IEEE Workshop on Information Forensics and Security (WIFS), pp. 1-6. IEEE, 2017.
```
@inproceedings{zhu2017latent,
  title={Latent fingerprint segmentation based on convolutional neural networks},
  author={Zhu, Yanming and Yin, Xuefei and Jia, Xiuping and Hu, Jiankun},
  booktitle={2017 IEEE Workshop on Information Forensics and Security (WIFS)},
  pages={1--6},
  year={2017},
  organization={IEEE}
}
```
3. Zhu, Yanming, Jiankun Hu, and Jinwei Xu. "A robust multi-constrained model for fingerprint orientation field construction." In 2016 IEEE 11th Conference on Industrial Electronics and Applications (ICIEA), pp. 156-160. IEEE, 2016.
```
@inproceedings{zhu2016robust,
  title={A robust multi-constrained model for fingerprint orientation field construction},
  author={Zhu, Yanming and Hu, Jiankun and Xu, Jinwei},
  booktitle={2016 IEEE 11th Conference on Industrial Electronics and Applications (ICIEA)},
  pages={156--160},
  year={2016},
  organization={IEEE}
}
```
4. Wang, Yi, Jiankun Hu, and Damien Phillips. "A fingerprint orientation model based on 2D Fourier expansion (FOMFE) and its application to singular-point detection and fingerprint indexing." IEEE Transactions on Pattern Analysis and Machine Intelligence 29, no. 4 (2007): 573-585.
```
@article{wang2007fingerprint,
  title={A fingerprint orientation model based on 2D Fourier expansion (FOMFE) and its application to singular-point detection and fingerprint indexing},
  author={Wang, Yi and Hu, Jiankun and Phillips, Damien},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={29},
  number={4},
  pages={573--585},
  year={2007},
  publisher={IEEE}
}
```
