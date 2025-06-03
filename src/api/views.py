from logging import Logger
from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    status,
    HTTPException,
    Form,
    Response,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.upload import upload_files
from src.api.schemas import (
    UploadFileSchemas,
    AskResponse,
    UserAddDB,
    UserAuth,
    EmailModel,
    UserInfo,
)
from src.api.dao import AuthDao
from src.api.auth_jwt import validate_auth_user, create_access_token
from src.api.dependencies import get_current_user
from src.core.db_helper import db_helper
from src.config import BASE_DIR, setup_logger
from src.chroma_db.chroma_storage import ChromaVectorStorage, get_chroma_storage
from src.chroma_db.llm_with_chroma_db import ChatWithLLM

logger: Logger = setup_logger(__name__)

router = APIRouter(prefix="/files", tags=["files"])


@router.post("/register/", status_code=status.HTTP_201_CREATED)
async def register_users(
    user: UserAddDB, session: AsyncSession = Depends(db_helper.get_session_with_commit)
) -> dict:
    dao = AuthDao(session)
    find_user = await dao.find_one_or_none(filters=EmailModel(email=user.email))
    if find_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Пользователь уже существует"
        )

    await dao.add(values=user)
    return {"message": "Вы успешно зарегистрированы!"}


@router.post("/login/")
async def auth_user(
    response: Response,
    user: UserAuth = Form(),
    session: AsyncSession = Depends(db_helper.get_session_without_commit),
) -> dict:

    check_user = await validate_auth_user(
        email=user.email, password=user.password, session=session
    )
    logger.info("Пользователь %s успешно авторизован", check_user)
    access_token = create_access_token(check_user)
    response.set_cookie(key="access_token", value=access_token, httponly=True)
    return {"ok": True, "access_token": access_token, "message": "Авторизация успешна!"}


@router.get("/me/")
async def get_me(user_data=Depends(get_current_user)) -> UserInfo:
    return UserInfo.model_validate(user_data)


@router.post("/logout/")
async def logout_user(response: Response):
    response.delete_cookie(key="access_token")
    return {"message": "Пользователь успешно вышел из системы"}


@router.post("/upload/")
async def files_upload(
    files: list[UploadFile] = File(...),
    user=Depends(get_current_user),
) -> list[UploadFileSchemas]:
    dir_for_upload: str = user.email
    upload_dir: Path = BASE_DIR / "upload_files" / dir_for_upload
    return await upload_files(files, upload_dir)


@router.post("/create_vector_storage/")
def create_vector_storage(
    user=Depends(get_current_user),
) -> dict[str, str]:
    name_storage: str = user.email
    vector_storage = ChromaVectorStorage(user_email=name_storage)
    vector_storage.create(
        folder_path=Path(BASE_DIR / "upload_files" / name_storage),
        persist_dir="src/chroma_db/vector_db",
    )
    return {"message": "Векторное хранилище создано"}


@router.delete("/delete_vector_storage/")
async def delete_vector_storage(
    user=Depends(get_current_user),
) -> dict:
    user_email: str = user.email
    vectorstore: ChromaVectorStorage = await get_chroma_storage(user_email=user_email)
    success = await vectorstore.delete_collection()
    if success:
        return {"message": f"Коллекция для {user_email} удалена"}
    else:
        raise HTTPException(status_code=400, detail="Не удалось удалить коллекцию")


@router.post("/ask/")
async def ask(
    query: AskResponse,
    user=Depends(get_current_user),
) -> dict[str, list]:
    user_email: str = user.email
    vectorstore: ChromaVectorStorage = await get_chroma_storage(user_email=user_email)

    results = await vectorstore.asimilarity_search(
        query=query.response, with_score=True, k=5
    )
    formatted_results: list = []

    for doc, score in results:
        formatted_results.append(
            {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": score,
            }
        )
    return {"results": formatted_results}


@router.post("/chat/")
async def chat_with_llm(
    query: AskResponse,
    user=Depends(get_current_user),
):
    user_email: str = user.email
    vectorstore: ChromaVectorStorage = await get_chroma_storage(user_email=user_email)

    results = await vectorstore.asimilarity_search(
        query=query.response, with_score=True, k=5
    )
    attachments = "\n".join([doc.page_content for doc, _ in results])
    llm = ChatWithLLM(model = "deepseek")
    respons = llm.response(query=query.response, attachments=attachments)
    return {"response": respons}
