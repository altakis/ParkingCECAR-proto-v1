import io
import gradio as gr
import matplotlib.pyplot as plt
import requests, validators
import torch
from PIL import Image
from transformers import (
    AutoFeatureExtractor,
    YolosForObjectDetection,
    DetrForObjectDetection,
)


# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def make_prediction(img, feature_extractor, model):
    inputs = feature_extractor(img, return_tensors="pt")
    outputs = model(**inputs)
    img_size = torch.tensor([tuple(reversed(img.size))])
    processed_outputs = feature_extractor.post_process(outputs, img_size)
    return processed_outputs[0]


def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    pil_img = Image.open(buf)
    basewidth = 750
    wpercent = basewidth / float(pil_img.size[0])
    hsize = int((float(pil_img.size[1]) * float(wpercent)))
    img = pil_img.resize((basewidth, hsize), Image.Resampling.LANCZOS)
    return img


def visualize_prediction(img, output_dict, threshold=0.5, id2label=None):
    keep = output_dict["scores"] > threshold
    boxes = output_dict["boxes"][keep].tolist()
    scores = output_dict["scores"][keep].tolist()
    labels = output_dict["labels"][keep].tolist()

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
    return fig2img(plt.gcf())


def reduce_visual_to_license(img, output_dict, threshold=0.5, id2label=None):
    keep = output_dict["scores"] > threshold
    boxes = output_dict["boxes"][keep].tolist()
    scores = output_dict["scores"][keep].tolist()
    labels = output_dict["labels"][keep].tolist()

    crop_img = img.crop(*boxes)
    return crop_img


def get_original_image(url_input):
    if validators.url(url_input):
        image = Image.open(requests.get(url_input, stream=True).raw)

        return image


def detect_objects(model_name, url_input, image_input, webcam_input, threshold):
    # Extract model and feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

    if "yolos" in model_name:
        model = YolosForObjectDetection.from_pretrained(model_name)
    elif "detr" in model_name:
        model = DetrForObjectDetection.from_pretrained(model_name)

    if validators.url(url_input):
        image = get_original_image(url_input)

    elif image_input:
        image = image_input

    elif webcam_input:
        image = webcam_input
        optional = True

    # Make prediction
    processed_outputs = make_prediction(image, feature_extractor, model)

    # Visualize prediction
    viz_img = visualize_prediction(
        image, processed_outputs, threshold, model.config.id2label
    )

    if optional:
        crop_img = reduce_visual_to_license(
            image, processed_outputs, threshold, model.config.id2label
        )
        return viz_img, crop_img
    return viz_img


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def set_example_url(example: list) -> dict:
    return gr.Textbox.update(value=example[0]), gr.Image.update(
        value=get_original_image(example[0])
    )


# Interface definition
import interface_options

demo = gr.Blocks(css=interface_options.css)

with demo:
    gr.Markdown(interface_options.title)
    gr.Markdown(interface_options.description)
    gr.Markdown(interface_options.twitter_link)
    options = gr.Dropdown(
        choices=interface_options.models,
        label="Object Detection Model",
        value=interface_options.models[0],
        show_label=True,
    )
    slider_input = gr.Slider(
        minimum=0.2, maximum=1, value=0.5, step=0.1, label="Prediction Threshold"
    )

    with gr.Tabs():
        with gr.TabItem("Image URL"):
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(
                        lines=2, label="Enter valid image URL here.."
                    )
                    original_image = gr.Image(shape=(750, 750))
                    url_input.change(get_original_image, url_input, original_image)
                with gr.Column():
                    img_output_from_url = gr.Image(shape=(750, 750))

            with gr.Row():
                example_url = gr.Examples(
                    examples=interface_options.urls, inputs=[url_input]
                )

            url_but = gr.Button("Detect")

        with gr.TabItem("Image Upload"):
            with gr.Row():
                img_input = gr.Image(type="pil", shape=(750, 750))
                img_output_from_upload = gr.Image(shape=(750, 750))

            with gr.Row():
                example_images = gr.Examples(
                    examples=interface_options.images, inputs=[img_input]
                )

            img_but = gr.Button("Detect")

        with gr.TabItem("WebCam"):
            with gr.Row():
                web_input = gr.Image(
                    source="webcam", type="pil", shape=(750, 750), streaming=True
                )
                img_output_from_webcam = gr.Image(shape=(750, 750))
                with gr.Column():
                    img_crop_from_webcam = gr.Image(shape=(750, 500))
                    gr.TextArea(
                        interactive=False,
                        label="license_text",
                        info="license character data",
                        lines=2,
                    )

            cam_but = gr.Button("Detect")

    url_but.click(
        detect_objects,
        inputs=[options, url_input, img_input, web_input, slider_input],
        outputs=[img_output_from_url],
        queue=True,
    )
    img_but.click(
        detect_objects,
        inputs=[options, url_input, img_input, web_input, slider_input],
        outputs=[img_output_from_upload],
        queue=True,
    )
    cam_but.click(
        detect_objects,
        inputs=[options, url_input, img_input, web_input, slider_input],
        outputs=[img_output_from_webcam, img_crop_from_webcam],
        queue=True,
    )

    gr.Markdown(
        "![visitor badge](https://visitor-badge.glitch.me/badge?page_id=nickmuchi-license-plate-detection-with-yolos)"
    )


demo.launch(debug=True, enable_queue=True)
