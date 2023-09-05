import io
import gradio as gr
from license_detector import license_detector
from FileManagerUtil import FileManagerUtil

detector = license_detector()


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def set_example_url(example: list) -> dict:
    return gr.Textbox.update(value=example[0]), gr.Image.update(
        value=detector.get_original_image(example[0])
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
        with gr.TabItem("WebCam"):
            with gr.Row():
                with gr.Column():
                    web_input = gr.Image(
                        source="webcam",
                        type="pil",
                        shape=(750, 750),
                        streaming=True,
                        mirror_webcam=False,
                    )
                img_output_from_webcam = gr.Image(shape=(750, 750))
                with gr.Column():
                    img_crop_from_webcam = gr.Image(shape=(500, 500))
                    gr.TextArea(
                        interactive=False,
                        label="license_text",
                        info="license character data",
                        lines=2,
                    )

            cam_but = gr.Button("Detect")

        with gr.TabItem("Image URL"):
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(
                        lines=2, label="Enter valid image URL here.."
                    )
                    original_image = gr.Image(shape=(750, 750))
                    url_input.change(
                        detector.get_original_image, url_input, original_image
                    )
                with gr.Column():
                    img_output_from_url = gr.Image(shape=(750, 750))
            with gr.Row():
                with gr.Column():
                    img_crop_from_url = gr.Image(shape=(500, 500))
                    gr.TextArea(
                        interactive=False,
                        label="license_text",
                        info="license character data",
                        lines=2,
                    )

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
                with gr.Column():
                    img_crop_from_upload = gr.Image(shape=(500, 500))
                    gr.TextArea(
                        interactive=False,
                        label="license_text",
                        info="license character data",
                        lines=2,
                    )

            with gr.Row():
                example_images = gr.Examples(
                    examples=interface_options.images, inputs=[img_input]
                )

            img_but = gr.Button("Detect")

    cam_but.click(
        detector.detect_objects,
        inputs=[options, url_input, img_input, web_input, slider_input],
        outputs=[img_output_from_webcam, img_crop_from_webcam],
        queue=True,
    )
    url_but.click(
        detector.detect_objects,
        inputs=[options, url_input, img_input, web_input, slider_input],
        outputs=[img_output_from_url, img_crop_from_url],
        queue=True,
    )
    img_but.click(
        detector.detect_objects,
        inputs=[options, url_input, img_input, web_input, slider_input],
        outputs=[img_output_from_upload, img_crop_from_upload],
        queue=True,
    )

    gr.Markdown(
        "![visitor badge](https://visitor-badge.glitch.me/badge?page_id=nickmuchi-license-plate-detection-with-yolos)"
    )


demo.launch(debug=True, enable_queue=True)
