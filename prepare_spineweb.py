import os
import os.path as path
import yaml
import torch
import numpy as np
import SimpleITK as sitk
import random
import shutil

from tqdm import tqdm
from PIL import Image
from adn.utils import read_dir, get_connected_components
from collections import defaultdict
from torchvision.utils import make_grid


def make_thumbnails(images):
    images = torch.tensor(np.array(images).astype(float))[:, np.newaxis, ...]
    images = (images - images.min()) / (images.max() - images.min())
    num_rows = int(len(images) ** 0.5)
    image = make_grid(
        images, nrow=images.shape[0] // num_rows, normalize=False)
    image = image.numpy().transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    return image


if __name__ == "__main__":
    config_file = "config/dataset.yaml"
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)['spineweb']

    patient_dirs = read_dir(
        config['raw_dir'], predicate=lambda x: "patient" in x, recursive=True)

    image_size = config['image_size']
    if type(image_size) is not list: image_size = [image_size] * 2
    thumbnail_size = config['thumbnail_size']
    if type(thumbnail_size) is not list: thumbnail_size = [thumbnail_size] * 2

    for patient_dir in tqdm(patient_dirs):
        patient_name = path.basename(patient_dir)
        volume_files = read_dir(patient_dir,
            predicate=lambda x: x.endswith("mhd") or x.endswith("nii.gz"), recursive=True)
        for volume_file in volume_files:
            volume_obj = sitk.ReadImage(volume_file)

            volume = sitk.GetArrayFromImage(volume_obj)
            volume_name = path.basename(volume_file).split(".")[0]

            thumbnails = defaultdict(list)
            index = 0
            for image in tqdm(volume,
                desc="Preparing {}_{}".format(patient_name, volume_name)):
                image_type = "no_artifact"

                # Check if the image has metal artifacts
                if image.max() > config["max_hu"][1]:
                    points = np.array(np.where(image > config["max_hu"][1])).T
                    points = set(tuple(p) for p in points)
                    components = get_connected_components(points)
                    max_area = max(len(c) for c in components)

                    if max_area > config["connected_area"]: image_type = "artifact"
                    else: continue
                elif image.max() > config["max_hu"][0]: continue

                output_dir = path.join(config["dataset_dir"], image_type,
                    "{}_{}".format(patient_name, volume_name))
                if not path.isdir(output_dir): os.makedirs(output_dir)

                image = Image.fromarray(image).resize(image_size)
                image = np.array(image)

                thumbnail = (image - image.min()) / (image.max() - image.min())
                thumbnail = (thumbnail * 255).astype(np.uint8)
                thumbnails[image_type].append(
                    np.array(Image.fromarray(image).resize(thumbnail_size)))

                image_name = "{}_{}_{:03d}".format(patient_name, volume_name, index)
                image_file = path.join(output_dir, image_name + ".npy")
                thumbnail_file = path.join(output_dir, image_name + ".png")
                
                np.save(image_file, image)
                Image.fromarray(thumbnail).save(thumbnail_file)
                index += 1

            # Create an overview of images from this patient
            for k, ts in thumbnails.items():
                output_dir = path.join(
                    config["dataset_dir"], k, "{}_{}".format(patient_name, volume_name))
                if len(ts) > 0:
                    thumbnails_file = path.join(config["dataset_dir"], k, "{}_{}.png".format(patient_name, volume_name))
                    Image.fromarray(make_thumbnails(ts)).save(thumbnails_file)
                else: os.removedirs(output_dir)

    # Create train and test split
    artifact_dir = path.join(config["dataset_dir"], "artifact")
    patient_dirs = read_dir(artifact_dir, "dir")
    random.shuffle(patient_dirs)

    test_patients = []
    test_cnt = 0
    index = 0
    while index < len(patient_dirs) and test_cnt < config["num_tests"]:
        num_images = len(read_dir(patient_dirs[index], "file"))
        if num_images < 100:
            test_patients.append(path.basename(patient_dirs[index]))
            test_cnt += num_images
        index += 1

    no_artifact_dir = path.join(config["dataset_dir"], "no_artifact")
    items = read_dir(artifact_dir) + read_dir(no_artifact_dir)

    test_dir = path.join(config["dataset_dir"], "test")
    train_dir = path.join(config["dataset_dir"], "train")
    if not path.isdir(test_dir): os.makedirs(test_dir)
    if not path.isdir(train_dir): os.makedirs(train_dir)

    for item in items:
        item_type, item_name = item.split(path.sep)[-2:]
        patient_name = path.splitext(item_name)[0]

        if patient_name in test_patients:
            shutil.move(item, path.join(test_dir, item_type, item_name))
        else:
            shutil.move(item, path.join(train_dir, item_type, item_name))

    shutil.rmtree(artifact_dir)
    shutil.rmtree(no_artifact_dir)
