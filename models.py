import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import ResNet, Bottleneck
import timm
from timm.models.layers.helpers import to_2tuple
from timm.models.vision_transformer import VisionTransformer
from transformers import ViTModel
import os
from abc import ABC, abstractmethod
from PathoDuet.vits import VisionTransformerMoCo
from RetCCL.ResNet import resnet50

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class AbstractModel(ABC, nn.Module):
    """
    Abstract base class for all models.

    Defines common methods like saving, loading.
    """
    def __init__(self):
        super(AbstractModel, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass


    def save(self, path, epoch, epochs_before_stop, best_val_loss, optimizer, scheduler, val_acc, training_time):
        """
        Save model parameters to a file.

        Args:
            path (str): File path to save the model.
            epoch (int): Current epoch.
            epochs_before_stop (int): Number of epochs before early stopping.
            best_val_loss (float): Best validation loss achieved so far.
            optimizer (torch.optim.Optimizer): Optimizer state.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler state.
            val_acc (float): Validation accuracy.
            training_time (float): Time taken for training.
        """
        state = {
            'epoch': epoch,
            'epochs_before_stop':epochs_before_stop,
            'best_val_loss':best_val_loss,
            'model': self.state_dict(),
            'optimizer': optimizer.state_dict() if optimizer is not None else None,
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'val_acc': val_acc,
            'training_time': training_time
        }
        torch.save(state, path+'.pth')
        return


    def load_model(self, path, optimizer, scheduler):
        """
        Load model parameters from a file.

        Args:
            path (str): File path to load the model from.
            optimizer (torch.optim.Optimizer): Optimizer object.
            scheduler (torch.optim.lr_scheduler._LRScheduler): Scheduler object.

        Returns:
            dict: Loaded model checkpoints.
        """
        checkpoints = torch.load(path + '.pth')
        self.load_state_dict(checkpoints['model'])
        if optimizer is not None and checkpoints['optimizer'] is not None:
            optimizer.load_state_dict(checkpoints['optimizer'])
        if scheduler is not None and checkpoints['scheduler'] is not None:
            scheduler.load_state_dict(checkpoints['scheduler'])
        return checkpoints



class ClassificationHead(AbstractModel):
    """
    Classification head consisting of fully connected layers.

    Args:
        input_size (int): Size of the input features.
        num_classes (int): Number of output classes.
    """

    def __init__(self, input_size=768, num_classes=2):
        super(ClassificationHead, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.relu1 = nn.ReLU() 
        self.fc2 = nn.Linear(512, 128)
        self.relu2 = nn.ReLU() 
        self.fc3 = nn.Linear(128, 64)
        self.relu3 = nn.ReLU() 
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x= self.relu1(x)
        x=self.fc2(x)
        x=self.relu2(x)
        x=self.fc3(x)
        x=self.relu3(x)
        x=self.fc4(x)
        return x
    

class ConvStem(nn.Module):
    """
    Convolutional stem for vision transformer models. Code from CTransPath (https://github.com/Xiyue-Wang/TransPath/tree/main)

    Args:
        img_size (int or tuple): Size of the input image.
        patch_size (int or tuple): Size of the image patch.
        in_chans (int): Number of input channels.
        embed_dim (int): Embedding dimension.
        norm_layer (nn.Module): Normalization layer.
        flatten (bool): Whether to flatten the output.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x



def get_ctranspath():
    """
    Load the ctranspath backbone model.

    Returns:
        nn.Module: Loaded backbone model.
    """
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    model.head = nn.Identity()
    td = torch.load(r'./backbones/ctranspath.pth')
    model.load_state_dict(td['model'], strict=True)
    return model


def get_phikon():
    """
    Load the phikon backbone model.

    Returns:
        nn.Module: Loaded backbone model.
    """
    model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)
    model.cuda()
    return model



def get_dino():
    """
    Load the Lunit-DINO backbone model.

    Returns:
        nn.Module: Loaded backbone model.
    """
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, num_heads=6, num_classes=0
    )
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get('DINO_p16')}"
    model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url))
    return model



def get_bt():
    """
    Load the Lunit-Barlow Twins backbone model.

    Returns:
        nn.Module: Loaded backbone model.
    """
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/bt_rn50_ep200.torch"),strict=True)
    return model



def get_swav():
    """
    Load the Lunit-SwAV backbone model.

    Returns:
        nn.Module: Loaded backbone model.
    """
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/swav_rn50_ep200.torch"),strict=False)
    return model



def get_moco():
    """
    Load the Lunit-MoCoV2 backbone model.

    Returns:
        nn.Module: Loaded backbone model.
    """
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3])
    model.load_state_dict(torch.hub.load_state_dict_from_url("https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights/mocov2_rn50_ep200.torch"),strict=False)
    return model



def get_retccl():
    """
    Load the RetCCL backbone model.

    Returns:
        nn.Module: Loaded backbone model.
    """
    model = resnet50(num_classes=128,mlp=False, two_branch=False, normlinear=True)
    pretext_model = torch.load(r'./backbones/retccl.pth')
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    return model



def get_pathoduet():
    """
    Load the PathoDuet backbone model.

    Returns:
        nn.Module: Loaded backbone model.
    """
    model = VisionTransformerMoCo(pretext_token=True, global_pool='avg')
    model.head = nn.Linear(768, 4)
    checkpoint = torch.load("./backbones/pathoduet.pth", map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    return model
    


def get_backbone(logger, name="ctranspath"):
    """
    Load a specific backbone model.

    Args:
        logger (logging.Logger): Logger object for logging information.
        name (str): Name of the backbone model to load.

    Returns:
        nn.Module: Loaded backbone model.
    """
    if name == "ctranspath":
        return get_ctranspath()
    elif name =="phikon":
        return get_phikon()
    elif name=="dino":
        return get_dino()
    elif name=="bt":
        return get_bt()
    elif name=="swav":
        return get_swav()
    elif name=="moco":
        return get_moco()
    elif name=="retccl":
        return get_retccl()
    elif name=="pathoduet":
        return get_pathoduet()
    else:
        logger.error("Wrong backbone")



class ResNetTrunk(ResNet):
    """
    ResNet trunk model.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class ResNet50(AbstractModel):
    """
    ResNet50 model for classification.

    Args:
        pretrained (bool, optional): Whether to use pretrained weights. Defaults to True.
        num_classes (int): Number of output classes. Defaults to 2.
    """
    def __init__(self, pretrained=True, num_classes=2):
        super(ResNet50, self).__init__()
        if pretrained:
            self.model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = torchvision.models.resnet50()
           
        #self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.model.fc = nn.Sequential(
            #nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.Linear(512,num_classes),
        )

    def forward(self, x):
        return self.model(x)
    
    def get_deep_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x
    