from pydantic import BaseModel


class ImageResponse(BaseModel):
    id: str
    inference_result: str
