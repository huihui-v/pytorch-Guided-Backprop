import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models
from torchvision import transforms as T


def load_img(imgpath):
    """Load image.
    
    Args:
        imgpath (string): The path of the image to load.
    Returns:
        ((int, int), torch.Tensor): The size of original image, and the normalized image tensor. 
    """
    img = Image.open(imgpath)
    ori_size = img.size
    
    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transforms(img)

    return (ori_size, img_tensor)

def save_img(grad, path):
    """Save result image.
    This function convert the gradient result into image, then write to file.
    Args:
        grad (torch.Tensor): The image tensor used by the model, shape (3, 224, 224).
        path (string): The output file path.

    Returns:
        None.
    """

    img = grad.detach().cpu().permute(1, 2, 0).numpy()
    img -= img.min()
    img /= img.max()
    img = np.uint8(255*img)

    cv2.imwrite(f"{path}", img)
    print(f"Save {path} complete")

class GBP(nn.Module):
    """An easy implementation of GBP, using a resnet bone.
    """
    def __init__(self):
        super(GBP, self).__init__()
        self.bone = models.resnet18(pretrained=True)
        self.bone.eval()

        self.set_backprop()

    def set_backprop(self):
        """Setting up backpropagation of Guided Backpropagation
        """
        # Hook function. Filter out all the negative gradients and pass through.
        def relu_backward_hook(module, grad_out, grad_in):
            modified_grad_out = nn.functional.relu(grad_out[0])
            return (modified_grad_out, )

        # Register the backward hook function for all ReLU layers.
        for idx, item in enumerate(self.bone.modules()):
            if isinstance(item, nn.ReLU):
                item.register_backward_hook(relu_backward_hook)


    def generate_gradient(self, input, target):
        # Forward through network
        input.requires_grad = True
        model_output = self.bone(input)
        self.bone.zero_grad()

        # Build initial gradient
        init_grad = torch.zeros_like(model_output).float()
        init_grad[0][target] = 1.

        # Backward through network
        model_output.backward(gradient=init_grad)

        # Return the gradient
        return input.grad

    def forward(self, input, target):
        return self.generate_gradient(input, target)


def main():
    img_id = "demo"
    class_idx = 243 # Mastiff

    img_path = os.path.join("data", img_id+'.png')

    _, img_tensor = load_img(img_path)

    model = GBP()
    img_tensor.unsqueeze_(0)
    grad = model(img_tensor, class_idx)

    save_img(grad[0], os.path.join("{}-{}.png".format(img_id, str(class_idx))))

main()