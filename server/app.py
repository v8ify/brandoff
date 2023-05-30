from typing import Union

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from image_service import get_blurred_image_path, get_replaced_image_path

import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/blur/")
async def create_upload_file(image: UploadFile):
    '''Accepts the image uploaded by user from the frontend and saves it with unique name
    in directory. The path of this file is then passed to object detection function.'''
    file_name = image.filename

    # extract the extension of file. e.g. .png, .jpeg, .jpg
    file_extension = os.path.splitext(file_name)[1]

    # create a unique file name using UUIDs
    unique_file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join("uploaded_images", unique_file_name)

    # write this image to a folder
    with open(file_path, "wb") as image_file:
        image_file.write(await image.read())
    
    blurred_image_path = get_blurred_image_path(file_path, file_extension)

    return FileResponse(blurred_image_path, content_disposition_type="attachment", media_type="application/octet-stream", filename=unique_file_name)

@app.post("/replace/")
async def create_upload_file(image: UploadFile):
    '''Accepts the image uploaded by user from the frontend and saves it with unique name
    in directory. The path of this file is then passed to object detection function.'''
    file_name = image.filename

    # extract the extension of file. e.g. .png, .jpeg, .jpg
    file_extension = os.path.splitext(file_name)[1]

    # create a unique file name using UUIDs
    unique_file_name = f"{uuid.uuid4()}{file_extension}"
    file_path = os.path.join("uploaded_images", unique_file_name)

    # write this image to a folder
    with open(file_path, "wb") as image_file:
        image_file.write(await image.read())
    
    replaced_image_path = get_replaced_image_path(file_path, file_extension)

    return FileResponse(replaced_image_path, content_disposition_type="attachment", media_type="application/octet-stream", filename=unique_file_name)

