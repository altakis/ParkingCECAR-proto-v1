from typing import List, Optional

from pydantic import BaseModel


class DetectionBase(BaseModel):
    original_image_name: str
    crop_image_name: str
    license_plate_data: str
    wall_time: float


class DetectionCreate(DetectionBase):
    pass


class Detection(DetectionBase):
    id: int

    class Config:
        orm_mode = True
