import base64
import copy
import json
from io import BytesIO
from typing import Tuple, Optional

import falcon
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from skimage.io import imsave
from torch import Tensor

from text_cosine_similarity import cosine_sim
from deeplabpytorch.demo import inference, preprocessing

# used at eval(self.CONFIG.MODEL.NAME)
from deeplabpytorch.libs.models import DeepLabV2_ResNet101_MSC


class TextImageSimilarity:
    """Class that computes how similar a text string is to an image"""

    def __init__(
        self,
        config_path: str = "deeplabpytorch/configs/cocostuff164k.yaml",
        model_path: str = "deeplabpytorch/deeplabv2_resnet101_msc-cocostuff164k-100000.pth",
        print_object_masked_images: bool = False,
        area_threshold: float = 0.01,
    ):
        """Initialize the Text-Image Similarity calculator

        Args:
            config_path: OmegaConf configuration file for the model
            model_path: Pre-trained model file location
            print_object_masked_images: Whether to store the output of the object
                detector in one annotated image per detected object
            area_threshold: Fraction of image size below with detected objects are ignored
        """
        self.print_object_masked_images = print_object_masked_images
        self.area_threshold = area_threshold
        self.setup_model_device_config_and_labels(config_path, model_path)

    def setup_model_device_config_and_labels(self, config_path, model_path):
        self.CONFIG = OmegaConf.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.set_grad_enabled(False)

        self.model = eval(self.CONFIG.MODEL.NAME)(
            n_classes=self.CONFIG.DATASET.N_CLASSES
        )
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

        label_file = self.CONFIG.DATASET.LABELS
        self.labels_no_spaces_dictionary = {}
        with open(label_file) as f:
            for line in f:
                key = line.split()[0]
                val = "".join(line.split()[1:])
                self.labels_no_spaces_dictionary[int(key)] = val

    def on_put(self, req, resp):
        text = req.get_param("text", required=True)
        imagedict_base64encoded = json.loads(req.stream.read())
        nparrayimage = decode_image(imagedict_base64encoded)
        resp = self.compute_similarity_and_repond(resp, nparrayimage, text)

    def compute_similarity_and_repond(self, resp, image, text):
        (
            similarity,
            output_labels,
        ) = self.detect_objects_in_image_compute_textual_similarity(image, text)
        return self.construct_response(resp, similarity, text, output_labels)

    def construct_response(
        self, resp, similarity: float, text: str, output_labels: Optional[str] = None
    ):
        """Constructs response for falcon. Optionally includes detected object labels"""
        payload = {"similarity": similarity}
        if output_labels:
            payload["debug_similarity"] = (
                "image[" + output_labels[:-1] + "] x " + text + " = " + str(similarity)
            )

        resp.body = json.dumps(payload)
        resp.status = falcon.HTTP_200
        return resp

    def detect_objects(self, image: Tensor) -> str:
        """Detects object in image and outputs their labels in a single
        concatenated string. Ignores small objects"""
        image, raw_image = preprocessing(image, self.device, self.CONFIG)
        labelmap = inference(self.model, image)
        labels = np.unique(labelmap)

        original_image = raw_image[:, :, ::-1]
        output_labels = ""
        for i, label in enumerate(labels):
            mask = labelmap == label
            area = sum(sum(mask))
            if area < mask.shape[0] * mask.shape[1] * self.area_threshold:
                continue
            output_labels += self.labels_no_spaces_dictionary[label] + " "

            if self.print_object_masked_images:
                notmask = labelmap != label
                this_label_image = copy.deepcopy(original_image)
                this_label_image[notmask] = 0
                imsave(
                    "area"
                    + str(area)
                    + "_"
                    + self.labels_no_spaces_dictionary[label]
                    + "["
                    + str(label)
                    + "].png",
                    this_label_image,
                    check_contrast=False,
                )
        return output_labels

    def detect_objects_in_image_compute_textual_similarity(
        self, image: Tensor, text: str
    ) -> Tuple[float, str]:
        """Detects objects in image bigger than area_threshold and computes
        textual similarity between the detected object labels and argument `text`

        Args:
            image: Input image in Torch tensor format
            text: Input text in string format

        Returns textual similarity as float and object labels concatenated in
        a single string"""
        output_labels = self.detect_objects(image)
        similarity_value = cosine_sim(output_labels, text)

        return similarity_value, output_labels


def decode_image(image_base64encoded):
    decoded_image = base64.b64decode(image_base64encoded.get("image"))
    PILimg = Image.open(BytesIO(decoded_image)).convert("L")
    return np.asarray(PILimg)
