import torch
from network import ViT
import torch
from transformers import ViTModel, ViTConfig
import torchvision

# import torch
# import torch.nn as nn
# from transformers import ViTModel, ViTConfig

# class AttentionMaskViT(ViTModel):
#     def __init__(self, config, target_patch_idx=20, attention_boost=10.0,regression_dim = 4*4*3):
#         super().__init__(config)
#         self.target_patch_idx = target_patch_idx  # 目标块索引
#         self.attention_boost = attention_boost    # 注意力权重增强系数

#         # 添加一个回归头，用于最终的回归任务输出
#         self.regression_head = nn.Linear(config.hidden_size, regression_dim)

#     def forward(self, pixel_values, attention_mask=None, **kwargs):
#         # 获取ViT的基础特征输出
#         outputs = self.embeddings(pixel_values)
        
#         # 获取批量大小
#         batch_size, num_patches, hidden_dim = outputs.size()
        
#         # 创建自定义的注意力掩码：在目标块处增加权重
#         custom_attention_mask = torch.ones((batch_size, num_patches, num_patches), device=outputs.device)
        
#         # 增强目标块的权重
#         custom_attention_mask[:, :, self.target_patch_idx] *= self.attention_boost
#         custom_attention_mask[:, self.target_patch_idx, :] *= self.attention_boost

#         # 调整每个ViT层的注意力计算
#         for i, layer_module in enumerate(self.encoder.layer):
#             # 在每层中引入自定义注意力掩码
#             layer_outputs = layer_module(outputs, attention_mask=custom_attention_mask)
#             outputs = layer_outputs[0]  # 提取第一个输出

#         # 取出CLS token的特征，用于回归头
#         cls_token_embedding = outputs[:, 0]  # 通常CLS token在第0个位置

#         # 通过回归头生成回归任务输出 (batch_size, 3)
#         regression_output = self.regression_head(cls_token_embedding)

#         # 将输出调整为 (batch_size, 3, 1)
#         regression_output = regression_output.unsqueeze(-1)
        
#         return regression_output

# # 初始化模型
# config = ViTConfig()
# model = AttentionMaskViT(config)

# # 输入示例
# pixel_values = torch.rand(1, 3, 224, 224)  # 批量大小为1的输入图像
# outputs = model(pixel_values)  # 输出形状为 (1, 3, 1)
# print(outputs.shape)  # 检查输出是否为 (batch_size, 3, 1)


# # 初始化模型
# config = ViTConfig()
# model = AttentionMaskViT(config)

# # 输入示例
# pixel_values = torch.rand(1, 3, 224, 224)  # 批量大小为1的输入图像
# outputs = model(pixel_values)

if __name__ == '__main__':
    image = torch.randn((2,3,32,32))
    patch = image[:,:,4:4+4,8:8+4]
    vit_test = ViT('ColorViT', pretrained=False,image_size=32,patches=4,num_layers=6,num_classes=4*4*3)
    out = vit_test(image,patch)
    print(out)

    from utils.utility import patch_split, patch_compose
    from PIL import Image
    import numpy as np
    real_image = Image.open('C:/Users/Administrator/Desktop/patch_sample.PNG').convert('RGB').resize((32,32))
    image_sample = torch.tensor(np.array(real_image)).permute(2,0,1).unsqueeze(0)/255.
    # image_sample = image_sample.cuda()
    image_patches = patch_split(image_sample)   # nP, C, H, W
    torchvision.utils.save_image(image_patches,
        "sample_batches_out.png",
        normalize=True,
        nrow=8,
        value_range=(0,1),    # update
    )
    image_recover = patch_compose(image_patches)
    torchvision.utils.save_image(image_recover,
        "sample_recover_out.png",
        normalize=True,
        nrow=1,
        value_range=(0,1),    # update
    )

