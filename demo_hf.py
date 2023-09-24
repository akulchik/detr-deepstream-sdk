from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw
import requests


url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

checkpoint = "facebook/detr-resnet-101"
processor = DetrImageProcessor.from_pretrained(checkpoint)
model = DetrForObjectDetection.from_pretrained(checkpoint)

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to COCO API
# let's only keep detections with score > 0.9
target_sizes = torch.tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

with Image.open(requests.get(url, stream=True).raw) as im:
    image_draw = ImageDraw.Draw(im)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        print(
                f"Detected {model.config.id2label[label.item()]} with confidence "
                f"{round(score.item(), 3)} at location {box}"
        )
        # Draw the bounding boxes on the image
        image_draw.rectangle(box, width=4)
        im.save("demo_hf.png", format="PNG")
