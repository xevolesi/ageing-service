import addict
import torch

from models.psp import pSp


CHECKPOINT_PATH = "backend/weights/model.pth"


# Augment checkpoint to run on CPU.
__checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
__checkpoint["opts"]["checkpoint_path"] = CHECKPOINT_PATH
__checkpoint["opts"]["input_nc"] = 4
__checkpoint["opts"]["start_from_encoded_w_plus"] = True
__checkpoint["opts"]["start_from_latent_avg"] = False
__checkpoint["opts"]["output_size"] = 1024


# Preload model once before service start.
model = pSp(addict.Dict(**__checkpoint["opts"]))
model.eval()
model.cpu()


@torch.no_grad()
def add_aging_channel(
    image_tensor: torch.FloatTensor, ages: list[int]
) -> torch.FloatTensor:
    ages = list(map(lambda age: int(age) / 100, ages))
    image_h, image_w = image_tensor.size()[-2:]
    aged_tensors = [
        torch.cat((image_tensor, age * torch.ones((1, 1, image_h, image_w))), dim=1)
        for age in ages
    ]
    return torch.cat(aged_tensors, dim=0)


@torch.no_grad()
def run_on_image(model: pSp, image_tensor: torch.FloatTensor) -> torch.FloatTensor:
    result = model(image_tensor, randomize_noise=False, resize=False)
    return result
