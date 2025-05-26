from fastapi import UploadFile, HTTPException
from pathlib import Path

from src.api.schemas import UploadFileSchemas


async def upload_files(
    files: list[UploadFile],
    upload_dir: Path,
) -> list[UploadFileSchemas]:
    saved_files: list = []
    upload_dir.mkdir(parents=True, exist_ok=True)

    for file in files:
        try:
            # Создаем путь для сохранения файла
            file_path = upload_dir / file.filename

            # Сохраняем файл
            with file_path.open("wb") as buffer:
                buffer.write(await file.read())

            saved_files.append(
                UploadFileSchemas(
                    filename=file.filename,
                    content_type=file.content_type,
                    size=file_path.stat().st_size,
                )
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Ошибка при загрузке файла {file.filename}: {str(e)}",
            )

    return saved_files
