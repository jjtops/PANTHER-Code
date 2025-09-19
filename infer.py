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
        
        subj = subjectsVal[1]

       #grid sampler used to create useful inferences
       #that can be combined to create a full size inference
        grid_sampler = tio.inference.GridSampler(
            subj,
            patch_size=(96, 96, 48),
            patch_overlap=(24, 24, 12) #standard value used 
        )
        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=4)
        aggregator = tio.inference.GridAggregator(grid_sampler)

        
        for patches_batch in patch_loader:
            img = patches_batch['image'][tio.DATA].float().to(device)
            locations = patches_batch[tio.LOCATION]
            img = img.permute(0, 1, 4, 2, 3)  #torchio creates a different shape then what the model will accept 
            logits = model(img)
            logits = logits.permute(0, 1, 3, 4, 2) 
            aggregator.add_batch(logits, locations)

        
        full_logits = aggregator.get_output_tensor().unsqueeze(0).to(device)

        
        full_logits = full_logits.permute(0, 1, 4, 2, 3)
        pred = torch.sigmoid(full_logits) > threshold
        pred = pred.permute(0, 1, 3, 4, 2)

        
        return pred.squeeze().cpu().numpy()




if __name__ == "__main__":
    subj = subjectsTrain[1]
    mask = subj['label'][tio.DATA].squeeze()

    info.interactive_slice_viewer_img(infer())

