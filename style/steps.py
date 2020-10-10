import utils

import torch
from torch.nn import functional as F

def disc_step(disc, opt, gen_img, style_img):
    disc.train()
    opt.zero_grad()

    d_real, _ = disc(style_img)
    d_gen, _ = disc(gen_img)

    if disc.mode == 'wass':
        # Wasserstein Distance
        dist = d_gen - d_real

        # Gradient Penalty
        x = utils.interpolate(gen_img, style_img)
        gp = disc.disc_gp(x)

        loss = dist + 10 * gp
    else:
        assert disc.mode == 'sn'
        # Spectral norm
        real_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        gen_loss = F.binary_cross_entropy_with_logits(d_gen, torch.zeros_like(d_gen))
        loss = real_loss + gen_loss

    loss.backward()

    opt.step()
    return loss.item()


def sc_step(model, opt, gen_img, args):
    model.eval()
    opt.zero_grad()

    if args.distance == "wass":
        disc_real, content_loss = model(gen_img)
        style_loss = -disc_real
    else:
        style_loss, content_loss = model(gen_img)

    loss = args.alpha * style_loss + (1 - args.alpha) * content_loss
    loss.backward()
    opt.step()

    return style_loss.item(), content_loss.item()
