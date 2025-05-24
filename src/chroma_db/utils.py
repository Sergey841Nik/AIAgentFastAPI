from pathlib import Path
from logging import Logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

from src.config import setup_logger

logger: Logger = setup_logger(__name__) 

def find_supported_files(folder_path: Path) -> list[Path]:
    """Находит все поддерживаемые файлы в указанной папке"""
    supported_extensions: list[str] = ['.pdf', '.docx', '.txt']
    files: list = []
    for ext in supported_extensions:
        files.extend(folder_path.glob(f'*{ext}'))
    return files

def document_generator(
    folder_path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 50
):
    """
    Генератор документов из всех файлов в папке
    Args:
        folder_path: Путь к папке с файлами
        chunk_size: Размер чанков в символах
        chunk_overlap: Перекрытие чанков
    Yield: 
        Документы LangChain с метаданными
    Raises:
        Exception Ошибка при обработке файла
    """
    if not folder_path.is_dir():
        raise ValueError(f"{folder_path} не является папкой")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for file_path in find_supported_files(folder_path):
        try:
            # Выбор лоадера по типу файла
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
                file_type = "PDF"
            elif file_path.suffix.lower() == '.docx':
                loader = Docx2txtLoader(str(file_path))
                file_type = "DOCX"
            elif file_path.suffix.lower() == '.txt':
                loader = TextLoader(str(file_path))
                file_type = "TXT"

            # Обработка документа
            for doc in loader.load():
                
                doc.metadata.update({
                    "source": str(file_path),
                    "file_type": file_type,
                    "file_name": file_path.name
                })
                
                yield from splitter.split_documents([doc])

            logger.info("✅ Обработан: %s", file_path.name)

        except Exception as e:
            logger.error("❌ Ошибка при обработке %s: %s", file_path.name, str(e))