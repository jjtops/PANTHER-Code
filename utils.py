import torch
import torch.nn as nn
from monai.metrics import DiceMetric
from torch.distributed.checkpoint import load_state_dict
import random
import monai
from model import UNet3D
import info





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# criterion = monai.losses.DiceLoss(sigmoid=True)

criterion = monai.losses.DiceCELoss(sigmoid=True, lambda_dice=1.0, lambda_ce=1.0)
criterion.to(device)


metric = monai.metrics.DiceMetric(include_background=False)

model = UNet3D()
# checkpoint_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)



optimizer = torch.optim.AdamW(model.parameters(), 1e-4)
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, min_lr=1e-6
)



epochs = 70

best_loss = float('inf')
best_score = -100000
random.seed(42)


t2w_mask1 = [path.replace('.mha', '') for path in info.t2w_mask]
t1w_mask1 = [path.replace('.mha', '') for path in info.t1w_mask]

shuffle_masks = random.sample(t2w_mask1, len(t2w_mask1))
train_masks = shuffle_masks[0:40]
val_masks = shuffle_masks[40:51]

shuffle_mask1 = random.sample(t1w_mask1, len(t1w_mask1))
train_mask1 = shuffle_mask1[0:68]
val_mask1 = shuffle_mask1[68:92]



# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])