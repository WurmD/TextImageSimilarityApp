import argparse
import json
import os

import cv2
import wget

from text_image_similarity import TextImageSimilarity, decode_image


def main():
    """Detects objects from image provided in argument, and computes textual
    similarity between its object labels and input text"""
    args = parse_arguments()
    text_image_sim = TextImageSimilarity(
        config_path=args.config_path, model_path=args.model_path
    )

    if args.image_path:
        image = cv2.imread(args.image_path, cv2.IMREAD_COLOR)
        (
            similarity_value,
            output_labels,
        ) = text_image_sim.detect_objects_in_image_compute_textual_similarity(
            image, args.textual_description
        )

    if args.imagedict_base64encoded:
        imagedict_base64encoded = json.loads(args.imagedict_base64encoded)
        nparrayimage = decode_image(imagedict_base64encoded)
        (
            similarity_value,
            output_labels,
        ) = text_image_sim.detect_objects_in_image_compute_textual_similarity(
            nparrayimage, args.textual_description
        )

    print(
        args.image_path + "[" + output_labels[:-1] + "] x",
        args.textual_description,
        "=",
        similarity_value,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compute similarity between image content and text description"
    )
    parser.add_argument(
        "--config_path", default="deeplabpytorch/configs/cocostuff164k.yaml", type=str
    )
    parser.add_argument(
        "--model_path",
        default="deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth",
        type=str,
    )
    parser.add_argument("--image_path", default="", type=str)
    parser.add_argument("--textual_description", default="", type=str)
    parser.add_argument("--imagedict_base64encoded", type=str)
    args = parser.parse_args()

    if (
        args.model_path
        == "deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth"
        and not os.path.exists(args.model_path)
    ):
        wget.download(
            "https://www.dropbox.com/s/icpi6hqkendxk0m/deeplabv2_resnet101_msc-cocostuff164k-100000.pth?raw=1",
            out="deeplabpytorch/",
        )

    return args
