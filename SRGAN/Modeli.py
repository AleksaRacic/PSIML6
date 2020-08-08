import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import vgg19
import math

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        vgg19_model = vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg19_model.features.children())[:18])

    def forward(self, img):
        return self.feature_extractor(img)



class ResBlock(nn.Module):
    def __init__(self, in_features):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)




class ResNetG(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_residual_blocks=8):  #8 je za SRResNet, a 16 za GAN
        super(ResNetG, self).__init__()

        # First layer
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4), nn.PReLU())

        # Residual blocks
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)

        # Second conv layer post residual blocks
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64, 0.8))

        # Upsampling layers
        upsampling = []
        for out_features in range(2):
            upsampling += [
                # nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 256, 3, 1, 1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU(),
            ]
        self.upsampling = nn.Sequential(*upsampling)

        # Final output layer
        self.conv3 = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        out1 = self.conv1(x)
        out = self.res_blocks(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv3(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.output_shape = (1,1,16)
        self.input_shape = input_shape
        in_channels, img_height, img_width = self.input_shape

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)   
        self.batch_norm2 = nn.BatchNorm2d(128)             
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)   
        self.batch_norm3 = nn.BatchNorm2d(128)  

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)   
        self.batch_norm4 = nn.BatchNorm2d(256)             
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)   
        self.batch_norm5 = nn.BatchNorm2d(256)             

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)   
        self.batch_norm6 = nn.BatchNorm2d(512)             
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)   
        self.batch_norm7 = nn.BatchNorm2d(512)   

        self.linear1 = nn.Linear(512 * img_height // 16 * img_width // 16, 1024)   
        self.linear2 = nn.Linear(1024, 1)          
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.batch_norm1(self.conv2(x)))
        x = F.leaky_relu(self.batch_norm2(self.conv3(x)))
        x = F.leaky_relu(self.batch_norm3(self.conv4(x)))
        x = F.leaky_relu(self.batch_norm4(self.conv5(x)))
        x = F.leaky_relu(self.batch_norm5(self.conv6(x)))
        x = F.leaky_relu(self.batch_norm6(self.conv7(x)))
        x = F.leaky_relu(self.batch_norm7(self.conv8(x)))

        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.linear1(x))
        output = torch.sigmoid(self.linear2(x))

        return output


# class Discriminator(nn.Module):
#     def __init__(self, input_shape):
#         super(Discriminator, self).__init__()

#         self.input_shape = input_shape
#         in_channels, in_height, in_width = self.input_shape

#         self.output_shape = (1,1)

#         def discriminator_block(in_filters, out_filters, first_block=False):
#             layers = []
#             layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
#             if not first_block:
#                 layers.append(nn.BatchNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
#             layers.append(nn.BatchNorm2d(out_filters))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         layers = []
#         in_filters = in_channels
#         for i, out_filters in enumerate([64, 128, 256, 512]):
#             layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
#             in_filters = out_filters
#         out_filters = 1024
#         linear_in = 16 * 512 * 16 * 8
#         layers.append(nn.Linear(linear_in, out_filters))
#         layers.append(nn.LeakyReLU(0.2, inplace=True))
#         layers.append(nn.Linear(out_filters, 1))
#         layers.append(nn.Sigmoid())
#         self.model = nn.Sequential(*layers)

#     def forward(self, img):
#         return self.model(img)




# class _conv(nn.Conv2d):
#     def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
#         super(_conv, self).__init__(in_channels = in_channels, out_channels = out_channels, 
#                                kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True)
        
#         self.weight.data = torch.normal(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), 0.02)
#         self.bias.data = torch.zeros((out_channels))
        
#         for p in self.parameters():
#             p.requires_grad = True
        

# class conv(nn.Module):
#     def __init__(self, in_channel, out_channel, kernel_size, BN = False, act = None, stride = 1, bias = True):
#         super(conv, self).__init__()
#         m = []
#         m.append(_conv(in_channels = in_channel, out_channels = out_channel, 
#                                kernel_size = kernel_size, stride = stride, padding = (kernel_size) // 2, bias = True))
        
#         if BN:
#             m.append(nn.BatchNorm2d(num_features = out_channel))
        
#         if act is not None:
#             m.append(act)
        
#         self.body = nn.Sequential(*m)
        
#     def forward(self, x):
#         out = self.body(x)
#         return out

# class discrim_block(nn.Module):
#     def __init__(self, in_feats, out_feats, kernel_size, act = nn.LeakyReLU(inplace = True)):
#         super(discrim_block, self).__init__()
#         m = []
#         m.append(conv(in_feats, out_feats, kernel_size, BN = True, act = act))
#         m.append(conv(out_feats, out_feats, kernel_size, BN = True, act = act, stride = 2))
#         self.body = nn.Sequential(*m)
        
#     def forward(self, x):
#         out = self.body(x)
#         return out

# class Discriminator(nn.Module):
    
#     def __init__(self, img_feat = 3, n_feats = 64, kernel_size = 3, act = nn.LeakyReLU(inplace = True), num_of_block = 3, patch_size = 256):
#         super(Discriminator, self).__init__()
#         self.act = act
        
#         self.conv01 = conv(in_channel = img_feat, out_channel = n_feats, kernel_size = 3, BN = False, act = self.act)
#         self.conv02 = conv(in_channel = n_feats, out_channel = n_feats, kernel_size = 3, BN = False, act = self.act, stride = 2)
        
#         body = [discrim_block(in_feats = n_feats * (2 ** i), out_feats = n_feats * (2 ** (i + 1)), kernel_size = 3, act = self.act) for i in range(num_of_block)]    
#         self.body = nn.Sequential(*body)
        
#         self.linear_size = ((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))
        
#         tail = []
        
#         tail.append(nn.Linear(self.linear_size, 1024))
#         tail.append(self.act)
#         tail.append(nn.Linear(1024, 1))
#         tail.append(nn.Sigmoid())
        
#         self.tail = nn.Sequential(*tail)
        
        
#     def forward(self, x):
        
#         x = self.conv01(x)
#         x = self.conv02(x)
#         x = self.body(x)        
#         x = x.view(-1, self.linear_size)
#         x = self.tail(x)
        
#         return x