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
    identity = nn.Identity()

    # Get CNN and parse it's layers
    if args.cnn.startswith('vgg19-bn'):
        relu = 2 * int(args.cnn.endswith('relu'))
        cnn = models.vgg19_bn(pretrained=args.pretrained).features.to(args.device).eval()
        style_layers = [norm, cnn[:1 + relu], cnn[1 + relu:8 + relu], cnn[8 + relu:15 + relu], cnn[15 + relu:28 + relu], cnn[28 + relu:41 + relu]]
        content_layers = [norm, cnn[:28 + relu]]
    elif args.cnn.startswith('vgg19'):
        relu = int(args.cnn.endswith('relu'))
        cnn = models.vgg19(pretrained=args.pretrained).features.to(args.device).eval()
        style_layers = [identity, nn.Sequential(norm, cnn[:1 + relu]), cnn[1 + relu:6 + relu], cnn[6 + relu:11 + relu],
                        cnn[11 + relu:20 + relu], cnn[20 + relu:29 + relu]]
        content_layers = [norm, cnn[:20 + relu]]
    elif args.cnn == 'resnet18':
        cnn = models.resnet18(pretrained=args.pretrained).to(args.device).eval()
        style_layers = [norm, cnn.conv1, nn.Sequential(cnn.bn1, cnn.relu, cnn.maxpool, cnn.layer1), cnn.layer2,
                        cnn.layer3, cnn.layer4]
        content_layers = [norm, cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool, cnn.layer1, cnn.layer2, cnn.layer3]
    else:
        raise Exception(f"Unrecognized cnn_arch argument: {args.cnn}")

    # Freeze parameters
    for param in cnn.parameters():
        param.requires_grad_(False)   
    
    # Subet of layers?
    if args.layers is not None:
        style_layers = style_layers[:1 + args.layers]

    # Remove inplace operations
    for layers in [style_layers, content_layers]:
        for layer in layers:
            for module in layer.modules():
                if hasattr(module, 'inplace'):
                    module.inplace = False

    return style_layers, content_layers
