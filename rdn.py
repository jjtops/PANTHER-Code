import os
import torch
import torchio as tio
import numpy as np
import SimpleITK as sitk
from ds import*

def infer(device='cuda', threshold = .3):
    checkpoint_path = os.path.join(os.path.dirname(__file__), "bestPre2.pth")
    model = utils.model
    model.eval()

    with torch.no_grad():
        # Get single subject
        subj = subjectsTrain[1]

        # Set up grid sampling
        grid_sampler = tio.inference.GridSampler(
            subj,
            patch_size=(96, 96, 48),
            patch_overlap=(24, 24, 12)
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
        aggregator = tio.inference.GridAggregator(grid_sampler)

        # Process patches
        for patches_batch in patch_loader:
            img = patches_batch['image'][tio.DATA].float().to(device)
            locations = patches_batch[tio.LOCATION]
            img = img.permute(0, 1, 4, 2, 3)  # Permute for model
            logits = model(img)
            logits = logits.permute(0, 1, 3, 4, 2)  # Permute back
            aggregator.add_batch(logits, locations)

        # Get full volume prediction
        full_logits = aggregator.get_output_tensor().unsqueeze(0).to(device)

        # Apply permutation and get prediction
        full_logits = full_logits.permute(0, 1, 4, 2, 3)
        pred = torch.sigmoid(full_logits) > threshold
        pred = pred.permute(0, 1, 3, 4, 2)

        # Return as numpy array
        return pred.squeeze().cpu().numpy()




if __name__ == "__main__":
    subj = subjectsTrain[1]
    mask = subj['label'][tio.DATA].squeeze()
    # print(mask.shape)
    # print(infer().shape)
    # print(infer())

    # info.interactive_slice_viewer_img(mask)
    info.interactive_slice_viewer_img(infer())

