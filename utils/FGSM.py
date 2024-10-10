import torch

def FGSM_attack(image, epsilon):
    grad = image.grad.data
    sgn = grad.sign()
    perturbed_image = image + epsilon * sgn
    perturbed_image = torch.clamp(perturbed_image, min=-1, max=1)
    return perturbed_image