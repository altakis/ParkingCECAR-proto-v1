from sqlalchemy.orm import Session

from . import models, schemas


class DetectionRepo:
    async def create(detection: schemas.DetectionCreate, db: Session):
        db_detection = models.Detection(
            original_image_name=detection.original_image_name,
            crop_image_name=detection.crop_image_name,
            license_plate_data=detection.license_plate_data,
            wall_time = detection.wall_time
            )
        with db() as session:
            session.add(db_detection)
            session.commit()
            session.refresh(db_detection)
        return db_detection

    def fetch_by_id(db: Session, _id: int):
        with db:
            return db.query(models.Detection).filter(models.Detection.id == _id).first()

    """ def fetch_by_name(db: Session, name: str):
        return db.query(models.Detection).filter(models.Detection.name == name).first() """

    def fetch_all(db: Session, skip: int = 0, limit: int = 100):
        with db:
            return sesdbsion.query(models.Detection).offset(skip).limit(limit).all()

    async def delete(db: Session, _id: int):
        with db:
            db_detection = db.query(models.Detection).filter_by(id=_id).first()
            db.delete(db_detection)
            db.commit()

    async def update(db: Session, detection_data):
        with db:
            db.merge(detection_data)
            db.commit()
