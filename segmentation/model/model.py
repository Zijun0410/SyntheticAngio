from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import torchvision

class UNet_SMP(smp.Unet):
    def __init__(self, backbone='resnet101', in_channels=1, attention=None):
        self.encoder = backbone
        super().__init__(encoder_name=self.encoder, in_channels=in_channels, 
            encoder_weights='imagenet', activation='sigmoid', 
            decoder_attention_type=attention) # 'scse'

class UNet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1, init_features = 32):
        super(UNet, self).__init__()
        
        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name = "enc1")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder2 = UNet._block(features, features * 2, name = "enc2")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder3 = UNet._block(features * 2, features * 4, name = "enc3")
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder4 = UNet._block(features * 4, features * 8, name = "enc4")
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.bottleneck = UNet._block(features * 8, features * 16, name = "bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size = 2, stride = 2)
        self.decoder4 = UNet._block((features* 8) * 2, features * 8, name = "dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size = 2, stride = 2)
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name = "dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name = "dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size = 2, stride = 2)
        self.decoder1 = UNet._block(features * 2, features, name = "dec1")
        
        self.conv = nn.Conv2d(in_channels = features, out_channels = out_channels, kernel_size = 1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim = 1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim = 1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.conv(dec1))


    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = features, 
                            kernel_size = 3, 
                            padding = 1, 
                            bias = False,
                        ),
                    ),
                    (name +"norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace = True)),
                    (
                        name + "conv2", 
                        nn.Conv2d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = 3,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features = features)),
                    (name + "relu2", nn.ReLU(inplace = True)),
                ]
            )
        )

# class MnistModel(BaseModel):
#     def __init__(self, num_classes=10):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, num_classes)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


# class DenseNet121_UNet(nn.Module):
#     def __init__(self, in_channels = 1, out_channels = 1, init_features = 32):
#         super(DenseNet121_UNet, self).__init__()
#         features = init_features
#         # DenseNet121 Encoder
#         self.densenet121 = torchvision.models.densenet121(pretrained=True)
        
#         self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 64, kernel_size = 7, stride = 2, padding = 3)
#         self.maxpool1 = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

#         self.denseblock1 = self.densenet121.features.denseblock1
#         self.bottleneck1 = nn.Conv2d(in_channels = 256, out_channels = 128, kernel_size = 1) 

#         self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

#         self.denseblock2 = self.densenet121.features.denseblock2
#         self.bottleneck2 = nn.Conv2d(in_channels = 512, out_channels = 256, kernel_size = 1)
        
#         self.denseblock3 = self.densenet121.features.denseblock3
#         self.bottleneck3 = nn.Conv2d(in_channels = 1024, out_channels = 512, kernel_size = 1)

#         self.denseblock4 = self.densenet121.features.denseblock4
#         self.bottleneck4 = nn.Conv2d(in_channels = 1024, out_channels = 1024, kernel_size = 1)

#         # Decoder
#         self.upsample4 = nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 2, stride = 2)
#         self.conv256A = nn.Conv2d(in_channels = 1024, out_channels = 256, kernel_size = 3, padding = 1)
#         self.conv256B = nn.Conv2d(in_channels = 256, out_channels = 256, kernel_size = 3, padding = 1)

#         self.upsample3 = nn.ConvTranspose2d(in_channels = 256, out_channels = 256, kernel_size = 2, stride = 2)
#         self.conv128A = nn.Conv2d(in_channels = 512, out_channels = 128, kernel_size = 3, padding = 1)
#         self.conv128B = nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 3, padding = 1)

#         self.upsample2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 2, stride = 2)
#         self.conv64A = nn.Conv2d(in_channels = 256, out_channels = 64, kernel_size = 3, padding = 1)
#         self.conv64B = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size = 3, padding = 1)

#         self.upsample1 = nn.ConvTranspose2d(in_channels = 64, out_channels = 64, kernel_size = 2, stride = 2)
#         self.conv32A = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, padding = 1)
#         self.conv32B = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, padding = 1)

#         self.upsample = nn.ConvTranspose2d(in_channels = 32, out_channels = 32, kernel_size = 2, stride = 2)
#         self.conv16A = nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1)
#         self.conv16B = nn.Conv2d(in_channels = 16, out_channels = 16, kernel_size = 3, padding = 1)

#         self.conv_final = nn.Conv2d(in_channels = 16, out_channels = out_channels, kernel_size = 3, padding = 1)

#     def forward(self, x):
#         # Encoder: x.shape = [16, 1, 512, 512]
#         d_conv1 = self.conv1(x) # d_conv1 = [16, 64, 256, 256] 
#         y = self.maxpool1(d_conv1) # y = [16, 64, 128, 128]
#         y = self.denseblock1(y) # y = [16, 256, 128, 128]
       
#         d_conv2 = self.bottleneck1(y) # d_conv2 = [16, 128, 128, 128] 
#         y = self.avgpool(d_conv2) # y = [16, 128, 64, 64] 
#         y = self.denseblock2(y) # y = [16, 512, 64, 64]
       
#         d_conv3 = self.bottleneck2(y) # d_conv3 = [16, 256, 64, 64]
#         y = self.avgpool(d_conv3) # y = [16, 256, 32, 32]
#         y = self.denseblock3(y) # y = [16, 1024, 32, 32]

#         d_conv4 = self.bottleneck3(y) # y = [16, 512, 32, 32]
#         y = self.avgpool(d_conv4) # y = [16, 512, 16, 16]
#         y = self.denseblock4(y) # y = [16, 1024, 16, 16]
#         enc = self.bottleneck4(y) # enc = [16, 1024, 16, 16]

#         # Decoder: enc.shape = [16, 1024, 16, 16] 
#         up4 = self.upsample4(enc) # up4 = [16, 512, 32, 32] 
#         concat4 = torch.cat((d_conv4, up4), dim = 1) # concat4 = [16, 1024, 32, 32]
#         z = self.conv256A(concat4) # z = [16, 256, 32, 32]
#         z = self.conv256B(z) # z = [16, 256, 32, 32]

#         up3 = self.upsample3(z) # up3 = [16, 256, 64, 64]
#         concat3 = torch.cat((d_conv3, up3), dim = 1) # concat3 = [16, 512, 64, 64]
#         z = self.conv128A(concat3) # z = [16, 128, 64, 64]
#         z = self.conv128B(z) # z = [16, 128, 64, 64]

#         up2 = self.upsample2(z) # up2 = [16, 128, 128, 128]
#         concat2 = torch.cat((d_conv2, up2), dim = 1) # concat2 = [16, 256, 128, 128]
#         z = self.conv64A(concat2) # z = [16, 64, 128, 128]
#         z = self.conv64B(z) # z = [16, 64, 128, 128] 

#         up1 = self.upsample1(z) # up1 = [16, 64, 256, 256]
#         concat1 = torch.cat((d_conv1, up1), dim = 1) # concat1 = [16, 128, 256, 256]
#         z = self.conv32A(concat1) # z = [16, 32, 256, 256]
#         z = self.conv32B(z) # z = [16, 32, 256, 256]

#         dec = self.upsample(z) # dec = [16, 32, 512, 512]
#         dec = self.conv16A(dec) # dec = [16, 16, 512, 512]
#         dec = self.conv16B(dec) # dec = [16, 16, 512, 512]

#         prediction = self.conv_final(dec) # prediction = [16, 1, 512, 512]

#         return torch.sigmoid(prediction)
