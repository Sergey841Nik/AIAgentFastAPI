from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from src.api.upload import upload_files
from src.api.schemas import UploadFileSchemas
from src.config import BASE_DIR

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload", response_model=list[UploadFileSchemas])
async def files_upload(files: list[UploadFile] = File(...)) -> list[UploadFile]:
    upload_dir: Path = BASE_DIR / "upload_files" / "test"
    return await upload_files(files, upload_dir)