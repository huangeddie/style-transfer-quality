import torch
import torchvision.models as models
from torch import nn


class Normalization(nn.Module):
    def __init__(self, device):
        super(Normalization, self).__init__()
        mean = torch.tensor([0.485, 0.456, 0.406], requires_grad=False)
        std = torch.tensor([0.229, 0.224, 0.225], requires_grad=False)
        self.mean = mean.to(device).view(-1, 1, 1)
        self.std = std.to(device).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


def get_layers(args):
    norm = Normalization(args.device)

    # Get CNN and parse it's layers
    if args.cnn == 'vgg19-bn':
        cnn = models.vgg19_bn(pretrained=args.pretrained).features.to(args.device).eval()
        style_layers = [norm, cnn[:1], cnn[1:8], cnn[8:15], cnn[15:28], cnn[28:41]]
        content_layers = [norm, cnn[:28]]
    elif args.cnn == 'vgg19':
        cnn = models.vgg19(pretrained=args.pretrained).features.to(args.device).eval()
        style_layers = [norm, cnn[:1], cnn[1:6], cnn[6:11], cnn[11:20], cnn[20:29]]
        content_layers = [norm, cnn[:20]]
    elif args.cnn == 'resnet18':
        cnn = models.resnet18(pretrained=args.pretrained).to(args.device).eval()
        style_layers = [norm, cnn.conv1,
                        nn.Sequential(cnn.bn1, cnn.relu, cnn.maxpool, cnn.layer1),
                        cnn.layer2, cnn.layer3, cnn.layer4]
        content_layers = [norm, cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool,
                          cnn.layer1, cnn.layer2, cnn.layer3]
    else:
        raise Exception(f"Unrecognized cnn_arch argument: {args.cnn}")

    if args.layers is not None:
        style_layers = style_layers[:1 + args.layers]

    return style_layers, content_layers
