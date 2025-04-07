import torch
from baseline import basemodel
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

def super_resolve_image(lr_image_path, model_path, output_path):
    # 加载模型
    model = basemodel()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 加载低分辨率图像
    lr_image = Image.open(lr_image_path).convert('RGB')
    lr_tensor = ToTensor()(lr_image).unsqueeze(0)  # 转换为张量并添加批量维度

    # 将图像输入模型
    with torch.no_grad():  # 关闭梯度计算
        sr_tensor = model(lr_tensor)

    # 将张量转换为图像
    sr_image = ToPILImage()(sr_tensor.squeeze(0)).convert('RGB')
    sr_image.save(output_path)  # 保存超分辨率图像

# 使用示例
lr_image_path = 'inputs/lr.jpg'  # 低分辨率图像路径
model_path = 'epochs/base_epoch_4_50.pth'  # 模型路径
output_path = 'outputs/sr_cnn.jpg'  # 输出图像路径
super_resolve_image(lr_image_path, model_path, output_path)