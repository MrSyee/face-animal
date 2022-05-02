import uuid
from typing import Dict

import os
import json
from glob import glob
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
    img_process = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ]
    )
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ]
    )
    input_img = img_process(image)
    image_tensor = preprocess(input_img).unsqueeze(0)

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

    # Save preprocessed input image in database
    save_dir = "./database"
    save_path = f"{save_dir}/{len(glob(os.path.join(save_dir, '*.jpg')))}.jpg"
    input_img.save(save_path)

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
