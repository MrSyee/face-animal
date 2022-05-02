from pydantic import BaseModel
from typing import Dict


class ImageResponse(BaseModel):
    id: str
    class_result: str
    prob: Dict[str, float]
