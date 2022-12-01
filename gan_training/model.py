from model_utils import *
import torchvision.models as models

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, depth=4):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.depth = depth

        init_n = 64
        self.inc = DoubleConv(n_channels, init_n)

        for iDep in range(depth):
            setattr(self, f'down{iDep+1}', 
                Down(init_n*2**iDep, init_n*2**(iDep+1)))
            # e.g. when the depth is 4, the following 'down' layers would be creates 
                # self.down1 = Down(64, 128)
                # self.down2 = Down(128, 256)
                # self.down3 = Down(256, 512)
                # self.down4 = Down(512, 1024)
            setattr(self, f'up{iDep+1}', 
                Up(init_n*2**(depth-iDep), init_n*2**(depth-iDep-1)))
            # e.g. when the depth is 4, the following 'up' layers would be creates 
                # self.up1 = Up(1024, 512)
                # self.up2 = Up(512, 256)
                # self.up3 = Up(256, 128)
                # self.up4 = Up(128, 64)

        self.outc = OutConv(init_n, n_classes)

    def forward(self, x):

        setattr(self, 'x0', self.inc(x)) # self.x0 = self.inc(x)
        depth = self.depth

        for iDep in range(depth):
            setattr(self, f'x{iDep+1}', 
                getattr(self, f'down{iDep+1}')(getattr(self, f'x{iDep}'))
                )
        # e.g. when the depth is 4, forward through the following 'down' layers  
            # x1 = self.down1(x0)
            # x2 = self.down2(x1)
            # x3 = self.down3(x2)
            # x4 = self.down4(x3)

        for iDep in range(depth):
            setattr(self, f'x{depth-iDep}', 
                getattr(self, f'up{iDep+1}')(
                    getattr(self, f'x{depth-iDep}'), 
                    getattr(self, f'x{depth-iDep-1}')
                    )
                )
        # e.g. when the depth is 4, forward through the following 'up' layers    
            # x3 = self.up1(x4, x3)
            # x2 = self.up2(x3, x2)
            # x1 = self.up3(x2, x1)
            # x0 = self.up4(x1, x0)

        logits = self.outc(self.x0)
        
        return logits
        

class ResNet18(object):
    """ResNet model with modified input & output layer"""
    def __init__(self, input_channel=1, output_dim=2):
        super().__init__()
        # extract fc layers features
        self.model = models.resnet18()
        num_features = self.model.fc.in_features     
        self.model.fc = nn.Linear(num_features, output_dim)
        if input_channel !=3:
            self.model.conv1 = nn.Conv2d(input_channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) 

    def get_model(self):
        return self.model

def count_parameters(model):
    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The total number of parameter is {total_param/1e6 :.2f} million.')

if __name__ == '__main__':
    generator = UNet(2, 1, 2)
    imput_tensor = torch.Tensor(5, 2, 512, 512)
    gen_output = generator(imput_tensor)
    discriminator = ResNet18().get_model()
    output = discriminator(discriminator)
    print(output.shape)