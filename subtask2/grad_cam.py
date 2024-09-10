import matplotlib.pyplot as plt
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms


def gard_cam(image_path, model, target_layers, transform, reshape=None):
    model.eval()

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image=image)["image"]
    input_batch = input_tensor.unsqueeze(0)

    if reshape is not None:
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)
    else:
        cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(3)]  # class index

    grayscale_cam = cam(input_batch, targets)
    grayscale_cam = grayscale_cam[0, :]
    img_pil = transforms.ToPILImage()(input_tensor.squeeze().cpu())

    plt.imshow(img_pil)
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
    from model import EVATiny
    from config import CFG
    from augmentation import inference_transform

    model = EVATiny()
    model.load_state_dict(
        torch.load(
            r"C:\workspace\FASHION-HOW\subtask2\check_points\Best\20240818_170120_862\Best_EVATiny.pth",
            map_location=CFG.DEVICE,
        )
    )
    model.to(CFG.DEVICE)

    target_layers = [model.backbone.blocks[-1].norm1]
    transform = inference_transform()
    path = r"C:\workspace\dataset\FashionHow\subtask2\train\1209\BO00111-1.jpg"
    gard_cam(path, model, target_layers, transform, reshape=True)
