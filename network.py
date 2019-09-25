import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
class R2plus1D(nn.Module):
    def __init__(self):
        super(R2plus1D, self).__init__()
        self.feature = r2plus1d_18(pretrained=False).cuda()
        self.classifier = nn.Linear(400, 1)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

class C3DNet(nn.Module):
    """The C3D Networks"""

    def __init__(self, pretrained=False):
        super(C3DNet, self).__init__()

        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm1 = nn.BatchNorm3d(32)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))


        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm2 = nn.BatchNorm3d(64)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3a = nn.BatchNorm3d(128)

        self.conv3b = nn.Conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm3b = nn.BatchNorm3d(128)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4a = nn.BatchNorm3d(256)

        self.conv4b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm4b = nn.BatchNorm3d(256)

        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm5a = nn.BatchNorm3d(256)

        self.conv5b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm5b = nn.BatchNorm3d(256)

        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        '''
        self.conv6a = nn.Conv3d(1024, 1024, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.norm6a = nn.BatchNorm3d(1024)
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))        


        self.conv7a = nn.Conv3d(1024, 512, kernel_size=(1, 1, 1), padding=(1, 1, 1))
        self.norm7a = nn.BatchNorm3d(512)
        self.pool7 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_6 = nn.Linear(512, 1024)
        self.fc_7 = nn.Linear(1024, 1024)
        self.fc_8 = nn.Linear(1024, 1)

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        
        self.fc8 = nn.Linear(4096, 1)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        #if pretrained:
        #    self.__load_pretrained_weights()

        #self.softmax = nn.Softmax(dim=1)
        '''
    def forward(self, x):
        x = self.relu((self.conv1(x)))
        # x = self.norm1(x)
        x = self.pool1(x)

        x = self.relu((self.conv2(x)))
        # x = self.norm2(x)
        x = self.pool2(x)

        x = self.relu((self.conv3a(x)))
        # x = self.norm3a(x)
        x = self.relu((self.conv3b(x)))
        # x = self.norm3b(x)
        x = self.pool3(x)

        x = self.relu((self.conv4a(x)))
        # x = self.norm4a(x)
        x = self.relu((self.conv4b(x)))
        # x = self.norm4b(x)
        x = self.pool4(x)

        x = self.relu((self.conv5a(x)))
        # x = self.norm5a(x)
        x = self.relu((self.conv5b(x)))
        # x = self.norm5b(x)
        # x = self.pool5(x)
        logits = nn.AdaptiveAvgPool3d(x)
        #print(x.shape)
        '''
        x = self.relu((self.conv6a(x)))
        # x = self.norm6a(x)
        x = self.pool6(x)

        x = self.relu((self.conv7a(x)))
        # x = self.norm7a(x)
        x = self.pool7(x)

        x = x.squeeze(2)
        x = self.avg_pool(x)
        x = x.view(-1, 512)
        x = self.relu(self.fc_6(x))
        x = self.dropout(x)
        x = self.relu(self.fc_7(x))
        x = self.dropout(x)
        logits = self.fc_8(x)
        # x = x.view(-1, 8192)
        # x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # x = self.dropout(x)

        # logits = self.fc8(x)
        '''
        return logits#torch.sigmoid(logits)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
            # Conv1
            "features.0.weight": "conv1.weight",
            "features.0.bias": "conv1.bias",
            # Conv2
            "features.3.weight": "conv2.weight",
            "features.3.bias": "conv2.bias",
            # Conv3a
            "features.6.weight": "conv3a.weight",
            "features.6.bias": "conv3a.bias",
            # Conv3b
            "features.8.weight": "conv3b.weight",
            "features.8.bias": "conv3b.bias",
            # Conv4a
            "features.11.weight": "conv4a.weight",
            "features.11.bias": "conv4a.bias",
            # Conv4b
            "features.13.weight": "conv4b.weight",
            "features.13.bias": "conv4b.bias",
            # Conv5a
            "features.16.weight": "conv5a.weight",
            "features.16.bias": "conv5a.bias",
            # Conv5b
            "features.18.weight": "conv5b.weight",
            "features.18.bias": "conv5b.bias",
            # fc6
            "classifier.0.weight": "fc6.weight",
            "classifier.0.bias": "fc6.bias",
            # fc7
            "classifier.3.weight": "fc7.weight",
            "classifier.3.bias": "fc7.bias",
        }

        p_dict = torch.load('c3d-pretrained.pth')
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.conv4a, model.conv4b,
         model.conv5a, model.conv5b, model.fc6, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k