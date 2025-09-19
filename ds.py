from torch.utils.data import Dataset
import SimpleITK as sitk
import torch
import utils
import torchio as tio
import pickle
import info



def normalize_func(tensor):
    return (tensor - 76.701) / 108.946 # stats for T1 Dataset 
    

norm = tio.Lambda(
    normalize_func,
    include=['image']
)


conserv_augments = tio.Compose([
    # Intensity Augmentations w low probabilities, don't want too much noise
    tio.RandomBiasField(p=0.2, coefficients=(0.0, 0.2), include=('image',)),
    tio.RandomGamma(p=0.2, log_gamma=(-0.1, 0.1), include=('image',)),
    tio.RandomNoise(p=0.15, mean=0, std=(0.0, 0.01), include=('image',)),


    tio.RandomFlip(axes=(0,), p=0.5),

    
    tio.RandomAffine(
        scales=(0.98, 1.02),
        degrees=(0, 1, 0),
        translation=(0, 0.5, 0),
        isotropic=False,
        image_interpolation='linear'
    ),
])



train_aug = tio.Compose([norm, conserv_augments])
val_aug = tio.Compose([norm])

class T2WDataset(Dataset):
    def __init__(self,  path, transform):
        self.path = path
        self.transform = transform

    def __len__(self):
        return len(self.path)

    def __getitem__(self, ind):
        path2 = [pth.replace('LabelsTr', 'ImagesTr') for pth in self.path]
        mask_pth = f'{self.path[ind]}.mha'
        img_pth = f'{path2[ind]}_0000.mha'



        #creates a TorchIO subject to be used and unpacked in a Queue
        subject = tio.Subject(
            image=tio.ScalarImage(img_pth),
            label=tio.LabelMap(mask_pth),
        )

        #the dataset includes 3 labels, 0: background 1:tumor 
        #2: pancreas, this creates a mask of only 2 labels
        lbl = subject['label'].data.clone()
        lbl = (lbl == 1).long()
        subject['label'].set_data(lbl)

        if self.transform:
            subject = self.transform(subject)

        return subject





#large boosting of labels as the dataset is heavily 
#unbalanced, with 99% background and >1% tumor
train_sampler = tio.LabelSampler(
    patch_size=(96, 96, 48), #creates sufficient samples per subject
    label_name='label',
    label_probabilities={0:0.1, 1:0.9}
)



set = T2WDataset (utils.train_mask1, train_aug)
subjectsTrain = tio.SubjectsDataset([set[i] for i in range(len(set))])
queue = tio.Queue(
    subjectsTrain,
    max_length=10,
    samples_per_volume=8,
    sampler=train_sampler,
    num_workers=0,
    shuffle_subjects=True,
    shuffle_patches=True
)

#uniform sampler is used to replicated frequencies that aren't
#as extreme as the ones created
val_sampler = tio.UniformSampler(patch_size=(96, 96, 48))

setVal = T2WDataset(utils.val_mask1, val_aug)
subjectsVal = tio.SubjectsDataset([setVal[i] for i in range(len(setVal))])
