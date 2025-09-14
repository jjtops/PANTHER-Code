import os
import torch
import torchio as tio
import pickle
import numpy as np
from pathlib import Path
import SimpleITK as sitk
from model import UNet3D


#-------------
# load model
#-------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")



state_dict_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
checkpoint = torch.load(state_dict_path, map_location=device)
model = UNet3D()
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
print("model loaded")


#-------------
# preprocessing
#-------------

landmarks_path = os.path.join(os.path.dirname(__file__), "landmarks.pkl")
with open(landmarks_path, "rb") as f:
    landmarks_dict = pickle.load(f)

infer_norm = tio.Compose([
    tio.HistogramStandardization({'image': landmarks_dict}),
    tio.RescaleIntensity(out_min_max=(0, 1), include=['image']),
])
print("preprocessing done")


#-------------
# infer func
#-------------

def inference(input_path, output_path):
    print(f"input: {input_path}")

    image = tio.ScalarImage(input_path)
    subject = tio.Subject(image=image)
    subject = infer_norm(subject)


    img_tensor = subject["image"][tio.DATA].float().to(device)
    print(f"img_tensor shape: {img_tensor.shape}")


    sbj_img = tio.Subject(image=tio.Image(tensor=img_tensor, type=tio.INTENSITY))
    sampler = tio.inference.GridSampler(
        sbj_img,
        patch_size=(128,128,96),
        patch_overlap=(32,32,24)
    )

    aggregator = tio.inference.GridAggregator(sampler, overlap_mode="hann")


    with torch.no_grad():
        for patches_batch in sampler:
            input_tensor = patches_batch['image'][tio.DATA].to(device)
            input_tensor = input_tensor.permute(0, 3, 2, 1)
            input_tensor = input_tensor.unsqueeze(0)
            # print(f"input_tensor shape : {input_tensor.shape}")


            logits = model(input_tensor)
            preds = torch.sigmoid(logits)

            loc = patches_batch[tio.LOCATION].unsqueeze(0)

            preds = preds.permute(0,1,4,3, 2)
            aggregator.add_batch(preds, loc)



    full_probs = aggregator.get_output_tensor()
    full_probs = full_probs.permute(0, 3, 2, 1)
    print(f"full probs shape {full_probs.shape}")
    pred_mask = (full_probs > 0.4).long()[0]
    pred_mask = pred_mask.cpu().numpy()

    print(pred_mask.shape)

    original_img = sitk.ReadImage(input_path)
    sitk_img = sitk.GetImageFromArray(pred_mask.astype(np.uint8))
    sitk_img.CopyInformation(original_img)
    # sitk_img = sitk.Cast(sitk_img, sitk.sitkUInt8)
    # sitk_img.SetOrigin(original_img.GetOrigin())
    # sitk_img.SetSpacing(original_img.GetSpacing())
    # sitk_img.SetDirection(original_img.GetDirection())




    # sitk_img.SetMetaData("descrip", "segmentation mask")
    # sitk_img.SetMetaData("aux_file", "")
    # sitk_img.SetMetaData("intent_name", "")
    # sitk_img.SetMetaData("scl_slope", "1")
    # sitk_img.SetMetaData("scl_inter", "0")



    output_path = Path(output_path)
    if not str(output_path).endswith(".mha"):
        output_path = output_path.with_suffix(".mha")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sitk.WriteImage(sitk_img, str(output_path))
    print(f"saved segmentation to: {output_path}")


    # writer = sitk.ImageFileWriter()
    # writer.SetFileName(str(output_path))
    # writer.SetImageIO("NiftiImageIO")
    #
    #     # Set compression
    # writer.SetUseCompression(True)
    #
    #     # Execute write
    # writer.Execute(sitk_img)

    # writer = sitk.ImageFileWriter()
    # writer.SetFileName(str(output_path))
    # writer.SetImageIO("NiftiImageIO")
    # writer.Execute(sitk_img)




inference("/home/bubbles/submission/test_input/160473e7-8cfa-4d01-a39b-3f30d29eac23.mha", "/home/bubbles/PycharmProjects/PythonProject1/out/mask1.mha")


# output_path = str(output_path)
# if not output_path.endswith(".nii.gz"):
#     output_path = str(Path(output_path).with_suffix(".nii.gz"))
#



  # output_path = str(output_path)
    # if not output_path.endswith(".nii.gz"):
    #     base = Path(output_path).stem  # strips extension cleanly
    #     output_path = str(Path(output_path).parent / f"{base}.nii.gz")
    #
    # sitk.WriteImage(sitk_img, output_path)