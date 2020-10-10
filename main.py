import argparse

parser = argparse.ArgumentParser(description='Style Transfer')

# Style Loss
parser.add_argument('--distance', type=str, default='disc-wass',
                    choices=['disc-wass', 'disc-sn', 'quad', 'linear', 'gauss', 'norm', 'gram'])
parser.add_argument('--samples', type=int, default=1024)

# Transfer
parser.add_argument('--steps', type=int, default=500, help='num training steps')
parser.add_argument('--imsize', type=int, default=224, help='image size')
parser.add_argument('--img-lr', type=float, default=2e-2,
                    help='learning rate for image pixels')
parser.add_argument('--disc-lr', type=float, default=2e-2,
                    help='learning rate for discriminators')
parser.add_argument('--alpha', type=float, default=0.2, help='alpha')
parser.add_argument('--device', choices=['cuda', 'cpu'], default='cpu')

# CNN
parser.add_argument('--cnn', type=str, default='vgg19-bn',
                    choices=['vgg19-bn', 'vgg19', 'resnet18', 'dense121'])
parser.add_argument('--layers', type=int, default=5)
parser.add_argument('--random', dest='pretrained', action='store_false')

# Images
parser.add_argument('--init-img', type=str, default='content',
                    choices=['content', 'random'])
parser.add_argument('--style', type=str)
parser.add_argument('--content', type=str, default=None)

# Output
parser.add_argument('--out-dir', type=str, default='out/')

import os
import utils
import transfer_model
from transfer_model import cnn
import style
import matplotlib.pyplot as plt


def run(args):
    # Images
    style_img, content_img, gen_img = utils.get_starting_imgs(args)

    # CNN layers
    style_layers, content_layers = cnn.get_layers(args)

    # Make model
    model = transfer_model.make(args, style_layers, content_layers, style_img, content_img)

    # Transfer
    losses_dict = style.transfer(args, gen_img, style_img, model)

    # Plot generated image and losses
    # Plot the images and losses
    plt.figure(figsize=(10, 10))
    plt.imshow(gen_img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0))
    loss_fig = utils.plot_losses(losses_dict)
    plt.show()

    # Save generated image and losses to disk
    utils.save_tensor_img(gen_img, os.path.join(args.out_dir, 'gen.png'))
    loss_fig.savefig(os.path.join(args.out_dir, 'losses.pdf'))
    print(f"Results saved to '{args.out_dir}'")


if __name__ == '__main__':
    args = parser.parse_args()
    run(args)
