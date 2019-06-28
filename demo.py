import os
import os.path as path
import argparse
import numpy as np
import torch
import scipy.io as sio
import matplotlib.pyplot as plt

from adn.models import ADNTest
from adn.utils import read_dir, get_config, update_config
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdd


# Path and google drive ID of pretrained models
CHECKPOINTS = {
    "spineweb":(
        "runs/spineweb/spineweb_39.pt",
        "1eF-6YTJYlVa7fVMk8n9yQssAqzrhLO1T"),
    "deep_lesion": (
        "runs/deep_lesion/deep_lesion_49.pt",
        "1NqZtEDGMNemy5mWyzTU-6vIAVIk_Ht-N")}
# ADN model specs
MODEL_SPECS = {
  "g_type": "adn",
  "d_type": "nlayer",
  "adn": {
    "input_ch": 1,
    "base_ch": 64,
    "num_down": 2,
    "num_residual": 4,
    "num_sides": 3,
    "down_norm": "instance",
    "res_norm": "instance",
    "up_norm": "layer",
    "fuse": True,
    "shared_decoder": False},
  "nlayer": {
    "input_nc": 1,
    "ndf": 64,
    "n_layers": 2,
    "norm_layer": "instance"}}


def get_checkpoint(model_type):
    """ Get checkpoint of the model. If the checkpoint does not exist,
        it will download the checkpoint from google drive directly.
    """
    assert model_type in CHECKPOINTS, f"{model_type} model not supported"
    model_path, gdrive_id = CHECKPOINTS[model_type]

    if not path.isfile(model_path):
        try:
            print(f"Pretrained model {model_path} not found!")
            gdd.download_file_from_google_drive(
                file_id=gdrive_id, dest_path=model_path, unzip=False, showsize=True)
        except Exception:
            print(f"Downloading {model_path} failed! Please download it from google drive directly.")
    return model_path


def normalize(data, minmax):
    """ Normalize input data to [-1, 1]
    """
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 2.0 - 1.0
    return data


def load_image(image_file, model_type):
    """ Load and preprocess an image for ADN
    """
    image = np.load(image_file)

    # Preprocess image according to model type (dataset type)
    # Basically, ADN expects input values ranging from -1 to 1
    if model_type == "spineweb":
        VALUE_RANGE = [-1000.0, 2000.0]
    elif model_type == "deep_lesion":
        MIUWATER = 0.192
        VALUE_RANGE = [0.0, 0.5]

        # Convert HU value to attenuation coefficient
        image[image < -1000] = -1000
        image = image / 1000 * MIUWATER + MIUWATER
    else: raise ValueError(f"Unsupported model type {model_type}!")

    image = normalize(image, VALUE_RANGE)
    image = torch.FloatTensor(image[np.newaxis, np.newaxis, ...])
    return image


def plot_image(output_file, img_low, pred_high, img_high, pred_low):
    """ Visualize the artifact reduction and artifact transfer results
    """

    plt.figure(figsize=(12, 12), frameon=False)
    fig, axes = plt.subplots(2, 2)
    
    images = [[img_low, pred_high], [img_high, pred_low]]
    titles = [["Image A (with artifact)", "Image A (artifact reduced)"],
        ["Image B (without artifact)", "Image B (with image A's artifact)"]]

    for i in range(2):
        for j in range(2):
            axes[i][j].imshow(images[i][j], vmin=0.0, vmax=1.0, cmap="gray")
            axes[i][j].axis('off')
            axes[i][j].set_title(titles[i][j])

    fig.savefig(output_file, frameon=False, bbox_inches='tight')
    plt.close("all")


def to_hu(*images):
    """ Convert the image values to HU values
    """
    MIUWATER = 0.192
    return [(img - MIUWATER) * 1000.0 / MIUWATER  for img in images]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This demo removes metal artifacts from sample images using pretrained ADN models")
    parser.add_argument("model_type", choices=["spineweb", "deep_lesion"],
        help="the name of the pretrained ADN model")
    parser.add_argument("--checkpoint", default=None,
        help="a path to the checkpoint of the pretrained model")
    parser.add_argument("--sample_dir", default="samples/",
        help="a folder that contains all sample images to be evaluated")
    parser.add_argument("--results_dir", default="results/",
        help="a folder that stores demo results")
    parser.add_argument("--no_gpu", action="store_true",
        help="if specified, use CPU only for the evaluation")
    args = parser.parse_args()

    # Create model
    model = ADNTest(**MODEL_SPECS)
    if not args.no_gpu: model.cuda()

    # Update model weights with checkpoint
    checkpoint = args.checkpoint if args.checkpoint else get_checkpoint(args.model_type)
    model.resume(checkpoint)

    # Get sample image files
    low_files = read_dir(
        path.join(args.sample_dir, args.model_type, "with_art"), "file")
    high_files = read_dir(
        path.join(args.sample_dir, args.model_type, "without_art"), "file")
    image_files = list(zip(low_files, high_files))

    for low_file, high_file in tqdm(image_files):
        img_low = load_image(low_file, args.model_type)
        img_high = load_image(high_file, args.model_type)

        low_name = path.basename(low_file)[:-4]
        high_name = path.basename(high_file)[:-4]

        # Artifact reduction and artifact transfer
        with torch.no_grad():
            model.evaluate(img_low, img_high)
            pred_high, pred_low = model.pred_lh, model.pred_hl

        # Convert image values to [0, 1.0]        
        images = (img_low, pred_high, img_high, pred_low)
        to_npy = lambda *xs: [x.detach().cpu().numpy()[0, 0] * 0.5 + 0.5 for x in xs]
        images = to_npy(*images)

        output_dir = path.join(args.results_dir, args.model_type, f"{low_name}_{high_name}")
        if not path.isdir(output_dir): os.makedirs(output_dir)

        # Plot and save the results
        plot_file = path.join(args.results_dir, args.model_type, f"{low_name}_{high_name}.png")
        plot_image(plot_file, *images)

        low_name = path.basename(low_file)[:-4]
        high_name = path.basename(high_file)[:-4]
        images = to_hu(*images)
        output_names = (
            f"low_{low_name}_origin.npy",
            f"low_{low_name}_artifact_reduced.npy",
            f"high_{high_name}_origin.npy",
            f"high_{high_name}_artifact_transferred.npy")
        for i in range(4):
            output_file = path.join(output_dir, output_names[i])
            np.save(output_file, images[i])
