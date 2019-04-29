import torch

class deviceSystem:

    def gpu_device(self, gpu=True):
        return torch.device('cuda' if gpu and torch.cuda.is_available() else 'cpu')