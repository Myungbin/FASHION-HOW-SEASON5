import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from torchvision import transforms


def gard_cam(image_path, model, target_layers, transform, reshape=None):
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = np.float32(image) / 255
    input_tensor = transform(image=image)["image"]
    input_batch = input_tensor.unsqueeze(0)

    if reshape is not None:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(4)] # class index

    grayscale_cam = cam(input_batch, targets)
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
    img_pil = transforms.ToPILImage()(input_tensor.squeeze().cpu())

    # plt.imshow(img_pil)
    plt.imshow(grayscale_cam, cmap="jet", alpha=0.5)
    plt.axis("off")
    plt.show()


def reshape_transform(tensor, height=16, width=16):
    # height, width = patch size
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result


if __name__ == "__main__":
    import torch
    from model import EVATiny224
    from config import CFG
    from augmentation import inference_transform
    model = EVATiny224()
    model.load_state_dict(
        torch.load(
            r"C:\workspace\FASHION-HOW\subtask1\check_points\Submission\20240815_182900_acc505\Best_EVATiny224.pth",
            map_location=CFG.DEVICE,
        )
    )
    model.to(CFG.DEVICE)
    
    target_layers = [model.backbone.blocks[-1].norm1]
    transform = inference_transform()
    path = r"C:\workspace\dataset\FashionHow\subtask1\val\1209/BO00015-1.jpg"
    gard_cam(path, model, target_layers, transform, reshape=True)
