from typing import Union

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from image_service import get_blurred_image_path

import pathlib

app = FastAPI()


@app.post("/upload/")
def upload_image(image: UploadFile):
    if image.filename:
        result_image_path = get_blurred_image_path(image.filename, pathlib.Path(image.filename).suffix)
        return FileResponse(result_image_path)
