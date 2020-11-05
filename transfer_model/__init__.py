import torch
from torch import nn

from transfer_model import layers, kernels


class TransferModel(nn.Module):
    def __init__(self, style_layers, style_img, distance, sample_size=None):
        """
        Initialize layers.

        Args:
            self: (todo): write your description
            style_layers: (list): write your description
            style_img: (str): write your description
            distance: (str): write your description
            sample_size: (int): write your description
        """
        super().__init__()

        self.disc_mode = None
        if distance.startswith('disc-'):
            self.layer_type = 'disc'
            self.disc_mode = distance.split('disc-')[-1]
        else:
            self.layer_type = 'kernel'

        # Style
        main = []
        style_feat = style_img
        for cnn_layer in style_layers:
            with torch.no_grad():
                style_feat = cnn_layer(style_feat)
                style_feat.requires_grad_(False)

            if self.layer_type == 'disc':
                main.append(layers.StyleLayerDisc(self.disc_mode, cnn_layer, style_feat.shape[1], sample_size))
            else:
                assert self.layer_type == 'kernel'
                assert sample_size is not None
                kernel = kernels.kernel_map[distance]
                main.append(layers.StyleLayerKernel(cnn_layer, style_feat, kernel,
                                                    sample_size))
        self.style = nn.Sequential(*main)

    def configure_content(self, content_layers, content_img):
        """
        Configure the content.

        Args:
            self: (todo): write your description
            content_layers: (todo): write your description
            content_img: (todo): write your description
        """
        # Content
        self.content = nn.Sequential(*content_layers)
        with torch.no_grad():
            self.content_feat = torch.tanh(self.content(content_img))
            self.content_feat.requires_grad_(False)

    def forward(self, img):
        """
        Forward computation

        Args:
            self: (todo): write your description
            img: (todo): write your description
        """
        if hasattr(self, 'content'):
            semantic_feat = torch.tanh(self.content(img))
            content_loss = torch.mean((semantic_feat - self.content_feat) ** 2)
        else:
            content_loss = torch.tensor(0.0)

        _, style_losses = self.style((img, []))
        ret = 0
        for out in style_losses:
            ret += out
        return ret / len(style_losses), content_loss

    def disc_gp(self, x):
        """
        Returns the value of the potential.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        gp_sum = 0
        for disc in self.style.children():
            x, gp = disc.disc_gp(x)
            gp_sum += gp
        return gp_sum / len(self.style)

    def conv_parameters(self):
        """
        Return a list of kernel layers.

        Args:
            self: (todo): write your description
        """
        params = []
        for disc_layer in self.style.children():
            params.extend(list(disc_layer.conv.parameters()))
        return params

    def disc_parameters(self):
        """
        Return a list of all parameters of the layers.

        Args:
            self: (todo): write your description
        """
        params = []
        for disc_layer in self.style.children():
            params.extend(list(disc_layer.disc.parameters()))
        return params


def make(args, style_layers, content_layers, style_img, content_img):
    """
    Creates an image from a list of layers.

    Args:
        style_layers: (todo): write your description
        content_layers: (todo): write your description
        style_img: (str): write your description
        content_img: (todo): write your description
    """
    # Initialize model
    model = TransferModel(style_layers, style_img, args.distance, args.samples).to(args.device)

    # Freeze CNN
    for params in model.conv_parameters():
        params.requires_grad = False

    # Configure Content
    if args.content is not None:
        model.configure_content(content_layers, content_img)

    return model
