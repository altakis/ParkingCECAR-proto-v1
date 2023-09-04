from sqlalchemy.orm import Session
import time
import asyncio

import sql_app.models as models
import sql_app.schemas as schemas
from db import get_db, engine, SessionLocal
from sql_app.repositories import DetectionRepo

models.Base.metadata.create_all(bind=engine)


async def create_detection(
    detection_request: schemas.DetectionCreate, db: Session = SessionLocal
):
    """
    Create an Detection and store it in the database
    """
    return await DetectionRepo.create(detection=detection_request, db=db)
