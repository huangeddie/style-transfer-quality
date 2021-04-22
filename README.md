## About
Performs style transfer by using a neural network to discriminate between the style image features and the generated image features. Results yield higher quality transfers than contemporary methods.
There is also a PyTorch implementation of this code in the `pytorch` branch. 

## Style representation comparisons
![style representation](imgs/style_rep.png)

## Example usage
See the `main.py` for all argument options.

#### Style representation
```python
python run.py --style_image=imgs/starry_night.jpg --imsize=512 --loss=wass 
```
Creates the style representation of the Starry Night painting using the unmixed Wasserstein distance between the features.
Improves mid-level textures of the style compared to typical second order methods.

```python
python run.py --style_image=imgs/starry_night.jpg --imsize=512 --disc_model=mlp 
```
Creates the style representation of the Starry Night painting using a multi-layer perceptron neural network to discriminate between the features.
Improves high-level textures of the style compared to typical second order methods.

#### Style transfer
```python
python run.py --style_image=imgs/la_muse.jpg --content_image=imgs/golden_gate.jpg --imsize=512 --loss=wass
```
Runs style transfer using the unmixed Wasserstein distance to match the features. 
The style image and content images are the La Muse painting and a picture of the golden gate bridge respectively. 

## Style losses
The code supports different types of style losses:
* `m1`: Mean square error between the means of the distribution
* `m1_m2`: Mean square error between the means and unmixed variances of the distributions
* `m1_covar`: Mean square error between the means and covariances of the distributions 
* `corawm2`: Mean square error between the mixed second raw moments of the distributions. 
This is equivalent to the original method of style transfer, which is the mean square error of the Gramian matrices of the distributions.
* `wass`: Unmixed Wasserstein distances between the distributions
* None: No loss is used. This is used when a neural network discriminator is used instead.

## Style discriminator
Set `disc_model=mlp` when you want to dynamically define the style loss with a neural network discriminator. 

## Abstract
Style transfer boils down to a distribution matching problem, where the generated image must match the feature 
distribution of the style image within the same hidden layers of the pretrained model. To that end, we propose using 
statistical moments as metrics for assessing distribution matching. Current style transfer methods match the feature 
distributions using second order statistics, which has two major limitations: 1.) they cannot match the third or higher 
order moments, 2.) they cannot match the non-linear relationships between the dimensions. 
We propose two new methods of style transfer that address both of these limitations respectively, 
and significantly increase the quality in the mid-level and high-level textures of the style transfer.

# Requirements
This code uses Python 3

### Required packages
* PyTorch
* Numpy
* Matplotlib
* Pillow
* TQDM
