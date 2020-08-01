# About
Performs style transfer by defining the style loss as the Wasserstein distance between the distribution of the the style image features and the generated image features.

# Abstract
Neural style transfer (NST) is a powerful image generation technique that uses a convolutional neural network (CNN) to merge the content of one image with the style of another. Contemporary methods of NST use first or second order statistics of the CNN's features to achieve transfers with relatively little computational cost. However, these methods cannot fully extract the style from the CNN's features. We present a new algorithm for style transfer that fully extracts the style from the features by redefining the style loss as the Wasserstein distance between the distribution of features. Thus, we set a new standard in style transfer quality. In addition, we state two important interpretations of NST. The first is a re-emphasis from Li et al., which states that style is simply the distribution of features. The second states that NST is a type of generative adversarial network (GAN) problem.

# Style Representation Comparions
![style representation](imgs/style_rep.png)

# Paper
Huang, Eddie and Sahil Gupta. “Style is a Distribution of Features.” (2020).

[Arxiv](https://arxiv.org/abs/2007.13010)

If you find this code useful in your research, please consider citing:
```
@article{huang2020style,
  title={Style is a Distribution of Features},
  author={Huang, Eddie and Gupta, Sahil},
  journal={arXiv preprint arXiv:2007.13010},
  year={2020}
}
```
