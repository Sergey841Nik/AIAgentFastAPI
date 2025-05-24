from pathlib import Path
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.config import setup_logger

logger = setup_logger(__name__)

class SaveFiles(BaseModel):
    text: str = Field(description="Содержимое файла")
    filename: str = Field(description="Имя файла")

@tool("save_file", args_schema=SaveFiles, return_direct=True)
def save_file(text: str, filename: str, files_path: Path = Path("./src/chroma_db/files_ai")):
    """
    Создаёт и сохраняет текст в файл с именем filename.
    Args:
        text: текст, который нужно сохранить
        filename: имя файла, в который нужно сохранить текст
    """
    logger.info("Сохранение файла %s", filename)
    try:
        if not files_path.is_dir():
            files_path.mkdir(parents=True, exist_ok=True)
        file_path: Path = files_path / filename
        file_path.write_text(text, encoding='utf-8')
        logger.info("Файл сохранен")
        return f"Файл сохранён в {file_path}"
    except Exception as e:
        logger.error("Ошибка при сохранении файла: %s", e)
        return f"Ошибка при сохранении файла: {str(e)[:100]}"


    
    
