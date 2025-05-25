from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends

from src.api.upload import upload_files
from src.api.schemas import UploadFileSchemas, AskResponse
from src.config import BASE_DIR
from src.chroma_db.chroma_storage import ChromaVectorStorage

router = APIRouter(prefix="/files", tags=["files"])

@router.post("/upload", response_model=list[UploadFileSchemas])
async def files_upload(files: list[UploadFile] = File(...)) -> list[UploadFile]:
    upload_dir: Path = BASE_DIR / "upload_files" / "test"
    return await upload_files(files, upload_dir)

@router.post("/ask")
async def ask(
    query: AskResponse,
):
    vectorstore = ChromaVectorStorage(name_vector_storage="my_collection_test")
    await vectorstore.init()
    results = await vectorstore.asimilarity_search(
        query=query.response, with_score=True, k=5
    )
    formatted_results = []
    for doc, score in results:
        formatted_results.append({
            "text": doc.page_content,
            "metadata": doc.metadata,
            "similarity_score": score,
        })
    return {"results": formatted_results}