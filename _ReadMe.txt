Latent fingerprint enhancement implemented using Pytorch=1.10, and test in Linux 4.18

hardware requirement: GPU

Step 1: Install a pytorch environment according to the requirements.txt

Step 2: Generate latent fingerprint textures using 'Cartoon-Texture Image Decomposition'
	-- code for 'Cartoon-Texture Image Decomposition': http://www.ipol.im/pub/art/2011/blmv_ct/
	-- the parameter sigma in this code was set to 6
	
Step 3: Latent enhancement using the FingerGAN
	-- according to your image location, adjust the related parameters in the function '_init_configure()' in the file ./experiment/latent_fingerprint_enhancement.py
	-- run latent_fingerprint_enhancement.py to obtain the enhanced fingerprints