from readline import get_completer_delims
import torchvision.models as vision_models
import torch, math


class CustomizedFCOS(object):
    """Customized FCOS model with modified classification head and transformations"""
    def __init__(self, num_class=2, min_size=512, max_size=512):
        super().__init__()
        self.num_class = num_class
        self.min_size = min_size
        self.max_size = max_size
        
    def get_model(self):

        self.model = vision_models.detection.fcos_resnet50_fpn(
            weights=vision_models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT,
            weights_backbone=vision_models.ResNet50_Weights.IMAGENET1K_V1)

        num_anchors = self.model.head.classification_head.num_anchors
        updated_cls_logits = torch.nn.Conv2d(
            self.model.backbone.out_channels, num_anchors*self.num_class, kernel_size=3, stride=1, padding=1)
        torch.nn.init.normal_(updated_cls_logits.weight, std=0.01)  # as per pytorch code
        torch.nn.init.constant_(updated_cls_logits.bias, -math.log((1 - 0.01) / 0.01)) 
        self.model.head.classification_head.cls_logits = updated_cls_logits
        self.model.head.classification_head.num_classes = self.num_class

        self.model.transform = vision_models.detection.transform.GeneralizedRCNNTransform(
            min_size=self.min_size, max_size=self.max_size, 
            image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225])
        
        return self.model

def count_parameters(model, model_name):
    total_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The total number of parameter in the {model_name} is {total_param/1e6 :.2f} million.')