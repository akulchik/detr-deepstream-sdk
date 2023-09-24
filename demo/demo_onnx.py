import numpy as np
import onnx
import onnxruntime as ort
# from optimum.onnxruntime import ORTModelForObjectDetection
from PIL import Image, ImageDraw
import requests
from transformers import DetrImageProcessor, DetrForObjectDetection


DETR_ONNX_DIR = "detr-resnet-101-onnx"


def main() -> None:
    processor = DetrImageProcessor.from_pretrained(DETR_ONNX_DIR)
    # model = ORTModelForObjectDetection.from_pretrained(DETR_ONNX_DIR)
    model = onnx.load(f"{DETR_ONNX_DIR}/model.onnx")
    # for input in model.graph.input:
    #     print(input.name, end=": ")
    #     # get type of input tensor
    #     tensor_type = input.type.tensor_type
    #     # check if it has a shape:
    #     if (tensor_type.HasField("shape")):
    #         # iterate through dimensions of the shape:
    #         for d in tensor_type.shape.dim:
    #             # the dimension may have a definite (integer) value or a symbolic identifier or neither:
    #             if (d.HasField("dim_value")):
    #                 print(d.dim_value, end=", ")  # known dimension
    #             elif (d.HasField("dim_param")):
    #                 print(d.dim_param, end=", ")  # unknown dimension with symbolic name
    #             else:
    #                 print("?", end=", ")  # unknown dimension with no name
    #     else:
    #         print("unknown rank", end="")
    #     print()

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt")["pixel_values"]
    # print(inputs)

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    print(f"{output_names = }")

    result = session.run(output_names, {input_name: inputs.numpy().astype(np.float32)})
    for output_name, output in zip(output_names, result):
        print(f"{output_name = }")
        print(f"{output.shape = }")

    # target_sizes = torch.tensor([image.size[::-1]])
    # results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # with Image.open(requests.get(url, stream=True).raw) as im:
    #     image_draw = ImageDraw.Draw(im)
    #     for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    #         box = [round(i, 2) for i in box.tolist()]
    #         print(
    #                 f"Detected {model.config.id2label[label.item()]} with confidence "
    #                 f"{round(score.item(), 3)} at location {box}"
    #         )
    #         # Draw the bounding boxes on the image
    #         image_draw.rectangle(box, width=4)
    #         im.save("demo_onnx.png", format="PNG")


if __name__ == "__main__":
    main()
