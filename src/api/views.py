from logging import Logger
from pathlib import Path

from fastapi import APIRouter, UploadFile, File

from src.api.upload import upload_files
from src.api.schemas import UploadFileSchemas, AskResponse
from src.config import BASE_DIR, setup_logger
from src.chroma_db.chroma_storage import ChromaVectorStorage
from src.chroma_db.llm_with_chroma_db import ChatWithLLM

logger: Logger = setup_logger(__name__)

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload")
async def files_upload(files: list[UploadFile] = File(...)) -> list[UploadFileSchemas]:
    upload_dir: Path = BASE_DIR / "upload_files" / "test"
    return await upload_files(files, upload_dir)

@router.post("/create_vector_storage")
def create_vector_storage() -> dict[str, str]:
    vector_storage = ChromaVectorStorage(name_vector_storage="my_collection_test")
    vector_storage.create(folder_path=Path(BASE_DIR / "upload_files" / "test"), persist_dir="src/chroma_db/vector_db")
    return {"message": "Векторное хранилище создано"}

@router.post("/ask")
async def ask(
    query: AskResponse,
)-> dict[str, list]:
    vectorstore = ChromaVectorStorage(name_vector_storage="my_collection_test")
    await vectorstore.init()
    results = await vectorstore.asimilarity_search(
        query=query.response, with_score=True, k=5
    )
    formatted_results: list = []

    for doc, score in results:
        formatted_results.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": score,
        })
    return {"results": formatted_results}

@router.post("/chat")
async def chat_with_llm(
    query: AskResponse,
):
    vectorstore = ChromaVectorStorage(name_vector_storage="my_collection_test")
    await vectorstore.init()
    results = await vectorstore.asimilarity_search(
        query=query.response, with_score=True, k=5
    )
    formatted_context = "\n".join([doc.page_content for doc, _ in results])
    logger.info("Контекст:\n%s", formatted_context)
  
    llm = ChatWithLLM()
    respons = llm.response(query=query.response, formatted_context=formatted_context)
    return {"response": respons}