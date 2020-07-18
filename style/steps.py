import utils

def disc_step(disc, opt, gen_img, style_img):
    disc.train()
    opt.zero_grad()

    # Wasserstein Distance
    d_real, _ = disc(style_img)
    d_gen, _ = disc(gen_img)
    disc_loss = d_gen - d_real

    # Gradient Penalty
    x = utils.interpolate(gen_img, style_img)
    gp = disc.disc_gp(x)

    loss = disc_loss + 10 * gp
    loss.backward()

    opt.step()
    return disc_loss.item(), gp.item()


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
