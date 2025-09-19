import numpy as np
from collections import Counter
import glob
import utils
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

all_imgs = glob.glob("/home/bubbles/GC/ImagesTr/*")
#T1W CE arterial phase MRI paths of TR images
t1w = glob.glob("/home/bubbles/GC/ImagesTr/*_0001_0000.mha")
# T2W MR_Linac MRI
t2w = [f for f in all_imgs if f not in t1w]

all_masks = glob.glob("/home/bubbles/GC/LabelsTr/*")
t1w_mask = glob.glob("/home/bubbles/GC/LabelsTr/*_0001.mha")
t2w_mask = [f for f in all_masks if f not in t1w_mask]




def dataset_info (path, plot=False):
    print(f'length of path is {len(path)}')
    raw_arr = sitk.ReadImage(path[0])
    np_arr = sitk.GetArrayFromImage(raw_arr)
    print(f'image size is {np_arr.shape}')

    if plot:
        plt.imshow(np_arr[np_arr.shape[0]//2], cmap='gray')
        plt.show()

#not all images are the same size, was required 
def size_info(path):
    tot_d = []
    tot_w = []
    tot_h = []
    for i in path:
        image = sitk.ReadImage(i)
        image_array = sitk.GetArrayFromImage(image)
        tot_d.append(image_array.shape[0])
        tot_w.append(image_array.shape[1])
        tot_h.append(image_array.shape[2])

    depth_counts = Counter(tot_d)
    width_counts = Counter(tot_w)
    height_counts = Counter(tot_h)


    print("Unique depths found:", len(depth_counts))
    for depth, count in depth_counts.items():
        print(f"Depth {depth}: {count} volumes")


    print("Unique width found:", len(width_counts))
    for width, count in width_counts.items():
        print(f"Width {width}: {count} nums")


    print("Unique width found:", len(height_counts))
    for height, count in height_counts.items():
        print(f"Height {height}: {count} nums")


def interactive_slice_viewer_img(image_array):
    image_array = np.array(image_array)
    image_array = np.squeeze(image_array)
    print(image_array.shape)
    image_array = image_array.transpose(2,1,0)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)

    # Initial slice
    slice_idx = image_array.shape[0] // 2
    im = ax.imshow(image_array[slice_idx], cmap='gray')
    ax.set_title(f'Slice {slice_idx}')

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, image_array.shape[0]-1,
                    valinit=slice_idx, valfmt='%d')

    def update(val):
        slice_num = int(slider.val)
        im.set_array(image_array[slice_num])
        ax.set_title(f'Slice {slice_num}')
        fig.canvas.draw()

    slider.on_changed(update)
    plt.show()


def interactive_slice_viewer(image_path):
    image = sitk.ReadImage(image_path)
    image_array = sitk.GetArrayFromImage(image)
    image_array = (image_array - 122.288) / 273.525
    # image_array = (image_array == 1).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25)

    # Initial slice
    slice_idx = image_array.shape[0] // 2
    im = ax.imshow(image_array[slice_idx], cmap='gray')
    ax.set_title(f'Slice {slice_idx}')

    # Slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Slice', 0, image_array.shape[0]-1,
                    valinit=slice_idx, valfmt='%d')

    def update(val):
        slice_num = int(slider.val)
        im.set_array(image_array[slice_num])
        ax.set_title(f'Slice {slice_num}')
        fig.canvas.draw()

    slider.on_changed(update)
    plt.show()


def count_labels_dataset(path_list):
    total_counts = Counter()

    for f in path_list:
        arr = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.int32)
        total_counts.update(arr.ravel().tolist())

    total_voxels = sum(total_counts.values())
    total_percentages = {lbl: (cnt / total_voxels) * 100
                         for lbl, cnt in total_counts.items()}

    return total_counts, total_percentages













