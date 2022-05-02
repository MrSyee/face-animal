from pydantic import BaseModel
from typing import Dict, List


class ImageResponse(BaseModel):
    id: str
    class_result: str
    prob: Dict[str, float]
