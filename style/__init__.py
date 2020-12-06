from torch import optim
from torchvision.transforms import functional as F
from tqdm import tqdm

from style import steps


def get_optimizers(model, gen_img, args):
    if args.opt == 'adam':
        OptClass = optim.Adam
        kwargs = {}
    else:
        assert args.opt == 'sgd'
        OptClass = optim.SGD
        kwargs = {'momentum': 0.9}

    # Convolutional parameters
    if 'gan' in args.distance:
        disc_opt = OptClass(model.disc_parameters(), lr=args.disc_lr, **kwargs)
    else:
        disc_opt = None

    # Image parameters for optimization
    img_opt = OptClass([gen_img.requires_grad_()], lr=args.img_lr, **kwargs)

    return img_opt, disc_opt


def transfer(args, gen_img, style_img, model):
    # Optimizers
    img_opt, disc_opt = get_optimizers(model, gen_img, args)

    # Losses
    style_losses, content_losses = [], []
    disc_losses = []

    # Train
    pbar = tqdm(range(0, args.steps), 'Style Transfer')
    gen_hist = []
    try:
        for i in pbar:
            if 'gan' in args.distance:
                # Optimize the discriminator
                disc_loss = steps.disc_step(model, disc_opt, gen_img, style_img)
                disc_losses.append(disc_loss)

            # Optimize over style and content
            style_loss, content_loss = steps.sc_step(model, img_opt, gen_img, args)
            style_losses.append(style_loss)
            content_losses.append(content_loss)

            # Clamp the values of updated input image
            gen_img.data.clamp_(0, 1)

            # Progress Bar
            pbar_str = f'Style: {style_losses[-1]:.3} Content: {content_losses[-1]:.3} '
            if 'gan' in args.distance:
                pbar_str += f'Disc: {disc_losses[-1]:.1f}'
            pbar.set_postfix_str(pbar_str)

            # GIF frame
            if i % args.gif_frame == 0:
                gen_hist.append(F.to_pil_image(gen_img.squeeze(0)))

    except KeyboardInterrupt:
        pass

    # Return losses
    loss_dict = {'style': style_losses}
    if args.content is not None:
        loss_dict['content'] = content_losses
    if 'gan' in args.distance:
        loss_dict['disc'] = disc_losses

    return loss_dict, gen_hist
