import io
import gradio as gr
import matplotlib.pyplot as plt
import requests, validators
import torch
import pathlib
from PIL import Image
from transformers import AutoFeatureExtractor, YolosForObjectDetection
import os


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933]
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
    wpercent = (basewidth/float(pil_img.size[0]))
    hsize = int((float(pil_img.size[1])*float(wpercent)))
    img = pil_img.resize((basewidth,hsize), Image.Resampling.LANCZOS) 
    return img


def visualize_prediction(img, output_dict, threshold=0.5, id2label=None):
    keep = output_dict["scores"] > threshold
    boxes = output_dict["boxes"][keep].tolist()
    scores = output_dict["scores"][keep].tolist()
    labels = output_dict["labels"][keep].tolist()
    if id2label is not None:
        labels = [id2label[x] for x in labels if x == 0]

    plt.figure(figsize=(50, 50))
    plt.imshow(img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, (xmin, ymin, xmax, ymax), label, color in zip(scores, boxes, labels, colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=color, linewidth=10))
        ax.text(xmin, ymin, f"{label}: {score:0.2f}", fontsize=60, bbox=dict(facecolor="yellow", alpha=0.8))
    plt.axis("off")
    return fig2img(plt.gcf())
    
def get_original_image(url_input):
    if validators.url(url_input):
        image = Image.open(requests.get(url_input, stream=True).raw)
        
        return image

def detect_objects(model_name,url_input,image_input,webcam_input,threshold):
    
    #Extract model and feature extractor
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    
    model = YolosForObjectDetection.from_pretrained(model_name)
    
    
    if validators.url(url_input):
        image = get_original_image(url_input)
         
    elif image_input:
        image = image_input
        
    elif webcam_input:
        image = webcam_input
    
    #Make prediction
    processed_outputs = make_prediction(image, feature_extractor, model)
    
    #Visualize prediction
    viz_img = visualize_prediction(image, processed_outputs, threshold, model.config.id2label)
    
    return viz_img
        
def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])

def set_example_url(example: list) -> dict:
    return gr.Textbox.update(value=example[0]), gr.Image.update(value=get_original_image(example[0]))


title = """<h1 id="title">License Plate Detection with YOLOS</h1>"""

description = """
YOLOS is a Vision Transformer (ViT) trained using the DETR loss. Despite its simplicity, a base-sized YOLOS model is able to achieve 42 AP on COCO validation 2017 (similar to DETR and more complex frameworks such as Faster R-CNN).
The YOLOS model was fine-tuned on COCO 2017 object detection (118k annotated images). It was introduced in the paper [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/abs/2106.00666) by Fang et al. and first released in [this repository](https://github.com/hustvl/YOLOS). 
This model was further fine-tuned on the [Car license plate dataset]("https://www.kaggle.com/datasets/andrewmvd/car-plate-detection") from Kaggle. The dataset consists of 443 images of vehicle with annotations categorised as "Vehicle" and "Rego Plates". The model was trained for 200 epochs on a single GPU.
Links to HuggingFace Models:
- [nickmuchi/yolos-small-rego-plates-detection](https://huggingface.co/nickmuchi/yolos-small-rego-plates-detection)
- [hustlv/yolos-small](https://huggingface.co/hustlv/yolos-small)  
"""

models = ["nickmuchi/yolos-small-rego-plates-detection","nickmuchi/yolos-small-license-plate-detection"]
urls = ["https://drive.google.com/uc?id=1j9VZQ4NDS4gsubFf3m2qQoTMWLk552bQ","https://drive.google.com/uc?id=1p9wJIqRz3W50e2f_A0D8ftla8hoXz4T5"]

twitter_link = """
[![](https://img.shields.io/twitter/follow/nickmuchi?label=@nickmuchi&style=social)](https://twitter.com/nickmuchi)
"""

css = '''
h1#title {
  text-align: center;
}
'''
demo = gr.Blocks(css=css)

with demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(twitter_link)
    options = gr.Dropdown(choices=models,label='Object Detection Model',value=models[0],show_label=True)
    slider_input = gr.Slider(minimum=0.2,maximum=1,value=0.5,step=0.1,label='Prediction Threshold')
    
    with gr.Tabs():
        with gr.TabItem('Image URL'):
            with gr.Row():
                with gr.Column():
                    url_input = gr.Textbox(lines=2,label='Enter valid image URL here..')
                    original_image = gr.Image(shape=(750,750))
                with gr.Column():
                    img_output_from_url = gr.Image(shape=(750,750))
                
            with gr.Row():
                example_url = gr.Dataset(components=[url_input],samples=[[str(url)] for url in urls])
            
            url_but = gr.Button('Detect')
     
        with gr.TabItem('Image Upload'):
            with gr.Row():
                img_input = gr.Image(type='pil',shape=(750,750))
                img_output_from_upload= gr.Image(shape=(750,750))
                
            with gr.Row(): 
                example_images = gr.Dataset(components=[img_input],
                                            samples=[[path.as_posix()] for path in sorted(pathlib.Path('images').rglob('*.j*g'))])
                                                   
                
            img_but = gr.Button('Detect')
            
        with gr.TabItem('WebCam'):
            with gr.Row():
                web_input = gr.Image(source='webcam',type='pil',shape=(750,750),streaming=True)
                img_output_from_webcam= gr.Image(shape=(750,750))

            cam_but = gr.Button('Detect')
            
    url_but.click(detect_objects,inputs=[options,url_input,img_input,web_input,slider_input],outputs=[img_output_from_url],queue=True)
    img_but.click(detect_objects,inputs=[options,url_input,img_input,web_input,slider_input],outputs=[img_output_from_upload],queue=True)
    cam_but.click(detect_objects,inputs=[options,url_input,img_input,web_input,slider_input],outputs=[img_output_from_webcam],queue=True)
    example_images.click(fn=set_example_image,inputs=[example_images],outputs=[img_input])
    example_url.click(fn=set_example_url,inputs=[example_url],outputs=[url_input,original_image])
    

    gr.Markdown("![visitor badge](https://visitor-badge.glitch.me/badge?page_id=nickmuchi-license-plate-detection-with-yolos)")

    
demo.launch(debug=True,enable_queue=True)