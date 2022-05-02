import uuid
from typing import Dict

import cv2
import numpy as np
import onnxruntime
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from backend.model import ImageResponse
from fastapi import APIRouter, File, UploadFile
from PIL import Image

router = APIRouter()


@router.get("/health")
def health() -> Dict[str, str]:
    return {"health": "ok"}


@router.post("/image", response_model=ImageResponse)
def process_image(image: UploadFile = File(...)):
    id_ = str(uuid.uuid4())
    image = Image.open(image.file)

    # preprocess
    trasnforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )
    image_tensor = trasnforms(image)
    image_tensor = image_tensor.unsqueeze(0)

    print(type(image))

    # inference 요청
    onnx_model = onnxruntime.InferenceSession("./model/resnet152.onnx")
    input_name = onnx_model.get_inputs()[0].name
    output_name = onnx_model.get_outputs()[0].name

    # onnx result
    y_pred = onnx_model.run(None, {input_name: image_tensor.cpu().numpy()})[0]
    y_pred = torch.from_numpy(y_pred)
    smax = nn.Softmax(dim=1)
    smax_out = smax(torch.from_numpy(np.array(y_pred)))[0]
    cat_prob = smax_out.data[0]
    dog_prob = smax_out.data[1]
    class_result = "cat" if cat_prob > dog_prob else "dog"
    print(class_result, type(class_result))

    # 결과 return
    response = {
        "id": id_,
        "class_result": class_result,
        "prob": {
            "cat_prob": float(cat_prob),
            "dog_prob": float(dog_prob),
        },
    }
    return response
