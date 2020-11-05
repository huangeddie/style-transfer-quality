import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def center_crop_square(im, size):
    """
    Crops an image to image.

    Args:
        im: (array): write your description
        size: (int): write your description
    """
    width, height = im.size  # Get dimensions

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def image_loader(image_name, imsize, device):
    """
    Convert an image to a tensor.

    Args:
        image_name: (str): write your description
        imsize: (int): write your description
        device: (todo): write your description
    """
    loader = transforms.Compose([transforms.Resize(imsize),
                                 transforms.ToTensor()])

    image = Image.open(image_name)
    image = center_crop_square(image, min(*image.size))

    # gen batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def get_starting_imgs(args):
    """
    Return an instance of a device.

    Args:
    """
    style_img = image_loader(args.style, args.imsize, args.device)

    if args.content is not None:
        content_img = image_loader(args.content, args.imsize, args.device)
    else:
        content_img = None

    if args.init_img == 'content' and args.content is not None:
        gen_img = content_img.clone()
    else:
        assert args.init_img == 'random' or args.content is None
        gen_img = torch.randn(style_img.data.size(), device=args.device)
        gen_img.data.clamp_(0, 1)

    return style_img, content_img, gen_img


def save_tensor_img(out_img, outpath):
    """
    Save tensor to disk.

    Args:
        out_img: (todo): write your description
        outpath: (str): write your description
    """
    out_img = out_img.cpu().clone()
    # remove the gen batch dimension
    out_img = out_img.squeeze(0)
    out_img = transforms.ToPILImage()(out_img)

    out_img.save(outpath)
    return outpath


def plot_losses(losses_dict):
    """
    Plot the loss function.

    Args:
        losses_dict: (dict): write your description
    """
    num_plts = len(losses_dict.keys())
    fig = plt.figure(figsize=(5 * num_plts, 4))
    plot_dims = (1, num_plts)
    for j, k in enumerate(losses_dict.keys()):
        plt.subplot(*plot_dims, 1 + j)
        y = losses_dict[k]
        x = np.arange(len(y))
        plt.plot(x, y)
        plt.title(f"{k} losses")
    return fig


def interpolate(x, y):
    """
    Interpolate the interpolation

    Args:
        x: (array): write your description
        y: (array): write your description
    """
    alpha = torch.rand(1) * torch.ones(x.size())
    alpha = alpha.to(x.device)

    return alpha * x.detach() + ((1 - alpha) * y.detach())


import torch.autograd as autograd


def calc_gradient_penalty(f, x):
    """
    Calculate the gradient of an objective function.

    Args:
        f: (array): write your description
        x: (todo): write your description
    """
    x.requires_grad_(True)

    y = f(x)

    grad_outputs = torch.ones(y.size()).to(y.device)
    gradients = autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs,
                              create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_k(*xs, k):
    """
    Return k k k k k k k k k k k k k.

    Args:
        xs: (int): write your description
        k: (int): write your description
    """
    if k is None or k <= 0:
        if len(xs) == 1:
            return xs[0]
        return xs

    n = len(xs[0])
    idcs = np.random.choice(n, min(k, n), replace=False)
    ret = []
    for x in xs:
        assert len(x) == n
        ret.append(x[idcs])

    if len(ret) == 1:
        return ret[0]
    return ret
