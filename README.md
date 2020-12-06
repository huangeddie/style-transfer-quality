## About
Performs style transfer by using a neural network to discriminate between the style image features and the generated image features. Results yield higher quality transfers than contemporary methods. 

## Style representation comparions
![style representation](imgs/style_rep.png)

## Example usage
See the `main.py` for all argument options.

#### Style representation
```python
python main.py --distance=wgan-gp --style=imgs/starry_night.jpg --device=cuda
```
Creates the style representation of the Starry Night painting using the discriminator from WGAN-GP to discriminate between the features.
Training is run on a GPU.

#### Style transfer
```python
python main.py --distance=wgan-gp --style=imgs/la_muse.jpg --content=imgs/golden_gate.jpg --device=cuda
```
Runs style transfer using the discriminator from WGAN-GP to discriminate between the features. 
The style image and content images are the La Muse painting and a picture of the golden gate bridge respectively. 
Training is run on a GPU.

## Style distances
The code supports different types of style distances:
> MMD stands for Maximum Mean Discrepancy
* `sngan`: Binary cross entropy using the spectral norm discriminator from SNGAN
* `wgan-gp`: Wasserstein distance implemented using WGAN-GP
* `wgan-sn`: Wasserstein distance using the spectral norm discriminator from SNGAN
* `quad`: MMD with the quadratic kernel
* `linear`: MMD with the linear kernel
* `gauss`: MMD with the Gaussian kernel
* `norm`: Square error between 1st order statistics, mean and standard deviation
* `gram`: Square error between the gramian matrices. This is the original method of NST by Gatys et. al.

## Abstract
We present a new algorithm for neural style transfer (NST) that fully extracts the style by dynamically generating the style loss with a neural network discriminator. The discriminator is constantly trained to discriminate between the style and generated feature distributions. 

Contemporary methods of NST use first or second order statistics for distribution discrimination. While these statistics are computationally cheap and fast, they cannot fully discriminate between the two image's feature distributions. Thus the generated image cannot be optimized to fully extract the style. 

Compared to contemporary methods, our method yields significantly higher quality style transfers. Our use of discriminators also suggests that NST can be viewed as a type of generative adversarial network (GAN).

# Requirements
This code uses Python 3

### Required packages
* PyTorch
* Numpy
* Matplotlib
* Pillow
* TQDM
