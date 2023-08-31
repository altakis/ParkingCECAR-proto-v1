import pathlib

title = """<h1 id="title">License Plate Detection with YOLOS</h1>"""

description = """
YOLOS is a Vision Transformer (ViT) trained using the DETR loss. Despite its simplicity, a base-sized YOLOS model is able to achieve 42 AP on COCO validation 2017 (similar to DETR and more complex frameworks such as Faster R-CNN).
The YOLOS model was fine-tuned on COCO 2017 object detection (118k annotated images). It was introduced in the paper [You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](https://arxiv.org/abs/2106.00666) by Fang et al. and first released in [this repository](https://github.com/hustvl/YOLOS). 
This model was further fine-tuned on the [Car license plate dataset]("https://www.kaggle.com/datasets/andrewmvd/car-plate-detection") from Kaggle. The dataset consists of 443 images of vehicle with annotations categorised as "Vehicle" and "Rego Plates". The model was trained for 200 epochs on a single GPU.
Links to HuggingFace Models:
- [nickmuchi/yolos-small-rego-plates-detection](https://huggingface.co/nickmuchi/yolos-small-rego-plates-detection)
- [hustlv/yolos-small](https://huggingface.co/hustlv/yolos-small)  
"""

models = ["nickmuchi/yolos-small-finetuned-license-plate-detection","nickmuchi/detr-resnet50-license-plate-detection", "nickmuchi/yolos-small-rego-plates-detection"]
urls = ["https://drive.google.com/uc?id=1j9VZQ4NDS4gsubFf3m2qQoTMWLk552bQ","https://drive.google.com/uc?id=1p9wJIqRz3W50e2f_A0D8ftla8hoXz4T5"]
images = [[path.as_posix()] for path in sorted(pathlib.Path('images').rglob('*.j*g'))]

twitter_link = """
[![](https://img.shields.io/twitter/follow/nickmuchi?label=@nickmuchi&style=social)](https://twitter.com/nickmuchi)
"""

css = '''
h1#title {
  text-align: center;
}
'''