import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse


class Frequency_based_Token_Selection(nn.Module):
    def __init__(self, keep, stride=16, quality_aware=True):
        super().__init__()
        self.DWT = DWTForward(J=4, wave='haar', mode='zero')
        self.IDWT = DWTInverse(wave='haar', mode='zero')
        self.keep = keep
        self.window_size = 16
        self.stride = stride
        self.quality_aware = bool(quality_aware)

    # Here, the show function can produce the Fig.4 in the paper.
    def show(self, x, writer=None, epoch=None, img_path=None, mode=1):
        x = x[:12]
        num_rows = 2  # Number of rows in the display grid
        num_cols = 6  # Number of columns in the display grid
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))
        if x[0].shape[0] == 3:
            x = x.permute(0, 2, 3, 1)
        for i in range(12):
            mask_2d = x[i].cpu().numpy().astype(np.float32)
            row = i // num_cols
            col = i % num_cols
            axes[row, col].imshow(mask_2d, cmap='bwr')
            axes[row, col].axis('off')
        plt.tight_layout()
        plt.show()
        if writer is not None:
            if mode == 1:
                writer.add_figure('FREQUENCY_After', fig, global_step=epoch)
            elif mode == 2:
                writer.add_figure('FREQUENCY_Before', fig, global_step=epoch)

    
    def mask(self, Inverse,window_size=16):
        batch_size, height, width = Inverse.size(0), Inverse.size(-2), Inverse.size(-1)
        Inverse = torch.mean(Inverse, dim=1)
        # create a tensor to store the count of non-zero elements
        count_tensor = torch.zeros((batch_size, height //self.stride, width // self.stride),
                                   dtype=torch.int, device=Inverse.device)
        # For each image in the batch
        for batch_idx in range(batch_size):
            image = Inverse[batch_idx]  # 获取当前图像
            # With a sliding window, unfold the image into a tensor
            unfolded = F.unfold(image.unsqueeze(0).unsqueeze(0), window_size, stride=self.stride)
            # Turns elements greater than 0 into binary, then sums to count the number of elements greater than 0
            count = unfolded.gt(0).sum(1)
            count = count.view(height // self.stride, width // self.stride)
            count_tensor[batch_idx] = count
            # Get the index of the maximum value of each image
        _, topk_indices = torch.topk(count_tensor.flatten(1), int(self.keep), dim=1)
        topk_indices = torch.sort(topk_indices, dim=1).values
        selected_tokens_mask = torch.zeros((batch_size, (height // self.stride) * (width // self.stride)),
                                           dtype=torch.bool, device=Inverse.device)
        selected_tokens_mask.scatter_(1, topk_indices, 1)
        return selected_tokens_mask

    def _modal_weights(self, tensors, quality_scores=None):
        B = tensors[0].size(0)
        count = len(tensors)
        if (not self.quality_aware) or quality_scores is None:
            return torch.full((B, count), 1.0 / count,
                              dtype=tensors[0].dtype, device=tensors[0].device)
        weights = quality_scores[:, :count].to(device=tensors[0].device, dtype=tensors[0].dtype)
        weights = weights.clamp_min(0.0)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

    @staticmethod
    def _weighted_sum(tensors, weights):
        out = 0
        for i, tensor in enumerate(tensors):
            view_shape = [tensor.size(0)] + [1] * (tensor.dim() - 1)
            out = out + tensor * weights[:, i].view(*view_shape)
        return out

    def forward(self, x, y, z, img_path, pattern='a', mode=None, writer=None,
                step=None, quality_scores=None):
        Ylx, Yhx = self.DWT(x)
        Yly, Yhy = self.DWT(y)
        # You can try to insert the self.show here to reproduce the Fig.4
        if z is not None:
            Ylz, Yhz = self.DWT(z)
            weights = self._modal_weights([Ylx, Yly, Ylz], quality_scores)
            low = self._weighted_sum([Ylx, Yly, Ylz], weights)
            high = [
                self._weighted_sum([Yhx[i], Yhy[i], Yhz[i]], weights)
                for i in range(len(Yhx))
            ]
        else:
            weights = self._modal_weights([Ylx, Yly], quality_scores)
            low = self._weighted_sum([Ylx, Yly], weights)
            high = [
                self._weighted_sum([Yhx[i], Yhy[i]], weights)
                for i in range(len(Yhx))
            ]

        Inverse = self.IDWT((low, high))
        selected_tokens_mask = self.mask(Inverse=Inverse, window_size=self.window_size)

        return selected_tokens_mask
