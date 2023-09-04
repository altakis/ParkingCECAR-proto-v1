from sqlalchemy import Column, Integer, String, Float

from db import Base


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    original_image_name = Column(String(250))
    crop_image_name = Column(String(250))
    license_plate_data = Column(String(200))
    wall_time = Column(Float)

    def __repr__(self):
        return (
            "DetectionModel(license_plate_data=%s, original_image_name=%s,crop_image_name=%s, wall_time=%s)"
            % (self.license_plate_data, self.original_image_name, self.crop_image_name, self.wall_time)
        )
