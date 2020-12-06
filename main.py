import argparse

parser = argparse.ArgumentParser(description='Style Transfer')

# Style Loss
parser.add_argument('--distance', type=str, default='wgan-gp',
                    choices=['wgan-gp', 'sngan', 'wgan-sn', 'quad', 'linear', 'gauss', 'norm', 'gram'])
parser.add_argument('--samples', type=int, default=1024,
                    help='number of features to sample from for each layer per training step. if set to 0, all features are used')

# Training
parser.add_argument('--steps', type=int, default=1000, help='num training steps')
parser.add_argument('--imsize', type=int, default=224, help='image size')
parser.add_argument('--img-lr', type=float, default=1e-2,
                    help='learning rate for image pixels')
parser.add_argument('--disc-lr', type=float, default=1e-2,
                    help='learning rate for discriminators')
parser.add_argument('--disc-l2', type=float, default=0,
                    help='weight decay')
parser.add_argument('--opt', choices=['adam', 'sgd'], default='adam', help='optimizer')
parser.add_argument('--alpha', type=float, default=0.2,
                    help='style-content balance ratio. larger values weigh style more. ' \
                         'if doing style representation (no content), then this value is ignored')
parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')

# CNN
parser.add_argument('--cnn', type=str, default='vgg19-bn',
                    choices=['vgg19-bn', 'vgg19-bn-relu', 'vgg19', 'vgg19-relu', 'resnet18', 'dense121'])
parser.add_argument('--layers', type=int, default=None, help='number of layers to. should be within [0, 5]')
parser.add_argument('--disc-hdim', type=int, default=256, help='dimension of the hidden layers in the discriminator')
parser.add_argument('--random', dest='pretrained', action='store_false')

# Images
parser.add_argument('--init-img', type=str, default='random',
                    help='how to initialize the generated image. can be one of [random, content, <path to image>]')
parser.add_argument('--style', type=str, help='path to style image')
parser.add_argument('--content', type=str, default=None, help='optional path to content image')

# Output
parser.add_argument('--out-dir', type=str, default='out/', help='directory to save all work')
parser.add_argument('--gif-frame', type=int, default=100,
                    help='interval to save the generated image w.r.t to the training steps to make a GIF of the style transfer')

import os
import utils
import arch
from arch import cnn
import style


def run(args):
    # Images
    style_img, content_img, gen_img = utils.get_starting_imgs(args)

    # CNN layers
    style_layers, content_layers = cnn.get_layers(args)

    # Make model
    model = arch.make_model(args, style_layers, content_layers, style_img, content_img)

    # Transfer
    losses_dict, gen_hist = style.transfer(args, gen_img, style_img, model)

    # Plot losses
    loss_fig = utils.plot_losses(losses_dict)

    # Save
    # Resized style
    utils.save_tensor_img(style_img, os.path.join(args.out_dir, 'style.png'))
    # Generated image
    utils.save_tensor_img(gen_img, os.path.join(args.out_dir, 'gen.png'))
    gen_hist[0].save(os.path.join(args.out_dir, 'gen.gif'), save_all=True, append_images=gen_hist[1:])
    # Losses
    loss_fig.savefig(os.path.join(args.out_dir, 'losses.png'))
    print(f"Results saved to '{args.out_dir}'")


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    run(args)
