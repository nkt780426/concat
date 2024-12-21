import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.iresnet import iresnet18, iresnet34


# Code anh lâm
class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


## Single backbone
class EmbeddingNet(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, len_embedding = 2):
        super(EmbeddingNet, self).__init__()

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.6)
        self.last_linear = nn.Linear(1792, len_embedding, bias=False)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.

        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.

        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        return x

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, X):
        anchor_tensors = X[:, 0, :, :]
        positive_tensors = X[:, 1, :, :]
        negative_tensors = X[:, 2, :, :]
        
        anchors = self.embedding_net(anchor_tensors)
        positives = self.embedding_net(positive_tensors)
        negatives = self.embedding_net(negative_tensors)
        
        return anchors, positives, negatives

    # X chỉ là tensor (batch_size, C, H, W) như bình thường
    def get_embedding(self, x):
        return self.embedding_net(x)
    
## Concat backbone
class EmbeddingNet_Concat(nn.Module):
    def __init__(self, conf):
        super(EmbeddingNet_Concat, self).__init__()
        # Load pre-trained ResNet models
        self.resnet1 = EmbeddingNet(conf['embedding_size'])
        self.resnet2 = EmbeddingNet(conf['embedding_size'])
        self.last_bn = nn.BatchNorm1d(2 * conf['embedding_size'], eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.logits = nn.Linear(1024, conf['embedding_size'], bias=True)
        
    def forward(self, X):
        normalmap_tensors = X[:, 0, :, :]
        albedo_tensors = X[:, 1, :, :]
        
        x1 = self.resnet1(normalmap_tensors)
        x1 = torch.flatten(x1, 1)
        
        x2 = self.resnet2(albedo_tensors)
        x2 = torch.flatten(x2, 1)        
        
        x = torch.cat((x1, x2), dim=1)
        x = self.last_bn(x)
        x = self.logits(x)
        
        return x
    

class TripletNet_Concat(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet_Concat, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, X):
        anchor_tensor = X[:, 0:2, :, :]
        positive_tensor = X[:, 2:4, :, :]
        negative_tensor = X[:, 4:6, :, :]
        
        output1 = self.embedding_net(anchor_tensor)
        output2 = self.embedding_net(positive_tensor)
        output3 = self.embedding_net(negative_tensor)
        return output1, output2, output3

    def get_embedding(self, X):
        return self.embedding_net(X)
    
        
## Code Hoang

class EmbeddingNet_Concat_V2(nn.Module):
    def __init__(self, num_classes=1000):
        super(EmbeddingNet_Concat, self).__init__()
        # Load pre-trained ResNet models
        self.resnet1 = iresnet18()
        self.resnet2 = iresnet18()
        self.resnet3 = iresnet34()
        self.last_bn = nn.BatchNorm1d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.logits = nn.Linear(1536, num_classes, bias=True)
        
    def forward(self, X):
        normalmap_tensors = X[:, 0, :, :]
        albedo_tensors = X[:, 1, :, :]
        depthmap_tensors = X[:, 2, :, :]
        
        # Forward pass for the first image
        normalmap_feature = self.resnet1(normalmap_tensors)
        normalmap_feature = torch.flatten(normalmap_feature, 1)
        
        albedo_feature = self.resnet2(albedo_tensors)
        albedo_feature = torch.flatten(albedo_feature, 1)   
        
        depthmap_feature = self.resnet3(depthmap_tensors)
        depthmap_feature = torch.flatten(depthmap_feature, 1)   
             
        # Concatenate the outputs
        x = torch.cat((normalmap_feature, albedo_feature, depthmap_feature), dim=1)
        
        # Apply batch normalization
        x = self.last_bn(x)
        
        # Final logits
        x = self.logits(x)
        
        return x
    
    
class TripletNet_Concat_V2(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet_Concat, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, X):
        anchor_tensors = X[:, 0:3, :, :]
        positive_tensors = X[:, 3:6, :, :]
        negative_tensors = X[:, 6:9, :, :]
        
        anchors = self.embedding_net(anchor_tensors)
        positives = self.embedding_net(positive_tensors)
        negatives = self.embedding_net(negative_tensors)
        
        return anchors, positives, negatives

    def get_embedding(self, x):
        return self.embedding_net(x)
    