import time
import asyncio
import io
import matplotlib.pyplot as plt
import requests, validators
import torch
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    YolosForObjectDetection,
    DetrForObjectDetection,
)
from FileManagerUtil import FileManagerUtil
import db_operator
import sql_app.schemas as schemas

# Initialize the OCR reader
import easyocr
import cv2
from numpy import asarray

reader = easyocr.Reader(["en"], gpu=False)

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


class license_detector:
    _models = [
        "nickmuchi/yolos-small-finetuned-license-plate-detection",
        "nickmuchi/detr-resnet50-license-plate-detection",
        "nickmuchi/yolos-small-rego-plates-detection",
    ]

    def __init__(self, model="") -> None:
        siu = FileManagerUtil()
        self.save_img_util = siu
        self._default_model = "nickmuchi/yolos-small-finetuned-license-plate-detection"

        if len(model) == 0:
            self._model = self._default_model
        else:
            self._model = model

    def getCurrentModel(self) -> str:
        return self._model

    def setModelName(self, model) -> None:
        self._model = model

    def verifyModel(self, model):
        if len(model) == 0:
            model = self._model
        else:
            self.setModelName(model)
        return self.getCurrentModel()

    def make_prediction(self, img, feature_extractor, model):
        inputs = feature_extractor(img, return_tensors="pt")
        outputs = model(**inputs)
        img_size = torch.tensor([tuple(reversed(img.size))])
        processed_outputs = feature_extractor.post_process(outputs, img_size)
        return processed_outputs[0]

    def fig2img(self, fig):
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        pil_img = Image.open(buf)
        basewidth = 750
        wpercent = basewidth / float(pil_img.size[0])
        hsize = int((float(pil_img.size[1]) * float(wpercent)))
        img = pil_img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
        return img

    def visualize_prediction(self, img, output_dict, threshold=0.5, id2label=None):
        keep = output_dict["scores"] > threshold
        boxes = output_dict["boxes"][keep].tolist()
        scores = output_dict["scores"][keep].tolist()
        labels = output_dict["labels"][keep].tolist()

        # Crops located license img for later ocr processing
        # crop_error values: 0 = None, 1 = cropping error, 2 = not found license
        crop_error = 0
        if len(boxes) > 0:
            try:
                crop_img = img.crop(*boxes)
            except:
                crop_error = 1
        else:
            crop_error = 2
            crop_img = img

        if id2label is not None:
            labels = [id2label[x] for x in labels]

        plt.figure(figsize=(50, 50))
        plt.imshow(img)
        ax = plt.gca()
        colors = COLORS * 100
        for score, (xmin, ymin, xmax, ymax), label, color in zip(
            scores, boxes, labels, colors
        ):
            if label == "license-plates":
                ax.add_patch(
                    plt.Rectangle(
                        (xmin, ymin),
                        xmax - xmin,
                        ymax - ymin,
                        fill=False,
                        color=color,
                        linewidth=10,
                    )
                )
                ax.text(
                    xmin,
                    ymin,
                    f"{label}: {score:0.2f}",
                    fontsize=60,
                    bbox=dict(facecolor="yellow", alpha=0.8),
                )
        plt.axis("off")

        license_located_img = self.fig2img(plt.gcf())
        if crop_error > 0:
            return license_located_img, license_located_img, crop_error
        return license_located_img, crop_img, crop_error

    def read_license_plate(self, license_plate_crop: Image.Image):
        # format PIL.Image input into grayscale
        license_plate_crop_gray = cv2.cvtColor(
            asarray(license_plate_crop), cv2.COLOR_BGR2GRAY
        )
        _, license_plate_crop_thresh = cv2.threshold(
            license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
        )

        detections = reader.readtext(license_plate_crop_gray)

        for detection in detections:
            bbox, text, score = detection

            text = text.upper().strip()

            return text, score

        return None, None

    def get_original_image(self, url_input):
        if validators.url(url_input):
            image = Image.open(requests.get(url_input, stream=True).raw)

            return image

    async def detect_objects(
        self, model_name, url_input, image_input, webcam_input, threshold
    ):
        # Time process
        start_time = time.perf_counter()

        model = self.verifyModel(model_name)

        # Extract model and feature extractor
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        if "yolos" in model_name:
            model = YolosForObjectDetection.from_pretrained(model_name)
        elif "detr" in model_name:
            model = DetrForObjectDetection.from_pretrained(model_name)

        if validators.url(url_input):
            image = self.get_original_image(url_input)

        elif image_input:
            image = image_input

        elif webcam_input:
            image = webcam_input
            # 'flipping' the vertical axis of the input may be needed
            # depending on configuration of the webcam and emulator
            # see Gradio (https://www.gradio.app/docs/image)
            # specially regarding mirror_webcam attribute
            # image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # Make prediction
        processed_outputs = self.make_prediction(image, feature_extractor, model)

        # Visualize prediction
        viz_img, crop_img, crop_error = self.visualize_prediction(
            image, processed_outputs, threshold, model.config.id2label
        )

        # save img results
        self.save_img_util.initialize_folders()
        img_ori_name, img_crop_name = self.save_img_util.save_img_results(
            viz_img, crop_img
        )

        # OCR license plate
        # TODO: OCR is too slow and frankly useless as the camera quality simply doesn't allow a good enough capture
        license_text, license_text_score = '', ''
        """ if crop_error == 0:
            license_text, license_text_score = self.read_license_plate(crop_img)
        else:
            license_text, license_text_score = "ERROR", 0 """

        # Time out and save to db
        process_time = time.perf_counter() - start_time
        data = schemas.DetectionBase(
            original_image_name=img_ori_name,
            crop_image_name=img_crop_name,
            license_plate_data=f"{license_text}:{license_text_score}",
            wall_time=process_time,
        )
        result = await db_operator.create_detection(detection_request=data)
        print(result)

        return viz_img, crop_img
