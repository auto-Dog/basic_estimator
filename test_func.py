import torch
from network import ViT

if __name__ == '__main__':
    image = torch.randn((2,3,32,32))
    patch = image[:,:,4:4+4,8:8+4]
    vit_test = ViT('ColorViT', pretrained=False,image_size=32,patches=4,num_layers=6,num_classes=4*4)
    out = vit_test(image,patch)
    print(out)