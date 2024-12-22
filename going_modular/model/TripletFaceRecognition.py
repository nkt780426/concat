import torch
import torch.nn as nn
from .backbone.iceptionresnetv1 import EmbeddingNet
from .backbone.iresnet import iresnet18, iresnet34


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
    def __init__(self, conf):
        super(EmbeddingNet_Concat_V2, self).__init__()
        # Load pre-trained ResNet models
        self.resnet1 = iresnet18(num_classes=conf['embedding_size'])
        self.resnet2 = iresnet18(num_classes=conf['embedding_size'])
        self.resnet3 = iresnet34(num_classes=conf['embedding_size'])
        self.last_bn = nn.BatchNorm1d(3*conf['embedding_size'], eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.logits = nn.Linear(3*conf['embedding_size'], conf['embedding_size'], bias=True)
        
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
        super(TripletNet_Concat_V2, self).__init__()
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
    