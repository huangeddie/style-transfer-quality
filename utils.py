import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


def center_crop_square(im, size):
    width, height = im.size  # Get dimensions

    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    return im


def image_loader(image_name, imsize, device):
    loader = transforms.Compose([transforms.Resize(imsize),
                                 transforms.ToTensor()])

    image = Image.open(image_name)
    image = center_crop_square(image, min(*image.size))

    # gen batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def get_starting_imgs(args):
    style_img = image_loader(args.style, args.imsize, args.device)

    if args.content is not None:
        content_img = image_loader(args.content, args.imsize, args.device)
    else:
        content_img = None

    if args.init_img == 'content':
        assert args.content is not None
        gen_img = content_img.clone()
    elif args.init_img == 'random':
        gen_img = torch.randn(style_img.data.size(), device=args.device)
        gen_img.data.clamp_(0, 1)
    else:
        gen_img = Image.open(args.init_img)
        gen_img = torch.as_tensor(np.asarray(gen_img), device=args.device)

    return style_img, content_img, gen_img


def save_tensor_img(out_img, outpath):
    out_img = out_img.cpu().clone()
    # remove the gen batch dimension
    out_img = out_img.squeeze(0)
    out_img = transforms.ToPILImage()(out_img)

    out_img.save(outpath)
    return outpath


def plot_losses(losses_dict):
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
    alpha = torch.rand(1) * torch.ones(x.size())
    alpha = alpha.to(x.device)

    return alpha * x.detach() + ((1 - alpha) * y.detach())


import torch.autograd as autograd


def calc_gradient_penalty(f, x):
    x.requires_grad_(True)

    y = f(x)

    grad_outputs = torch.ones(y.size()).to(y.device)
    gradients = autograd.grad(outputs=y, inputs=x, grad_outputs=grad_outputs,
                              create_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def sample_k(*xs, k):
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
