from logging import Logger

from fastapi import Request, HTTPException, status, Depends
from jwt.exceptions import InvalidTokenError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.models import User
from src.core.db_helper import db_helper
from src.api.utils import decoded_jwt
from src.api.dao import AuthDao
from src.api.schemas import EmailModel
from src.config import setup_logger


logger: Logger = setup_logger(__name__)


def get_token(request: Request) -> str:
    token = request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Токен истек"
        )
    return token


async def get_current_user(
    token: str = Depends(get_token),
    session: AsyncSession = Depends(db_helper.get_session_without_commit),
) -> User | None:
    try:
        payload = decoded_jwt(token)
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Токен не валидный"
        )

    user_email: str = payload.get("sub")
    dao = AuthDao(session)
    user: User | None = await dao.find_one_or_none(filters=EmailModel(email=user_email))

    logger.info("Найден пользователь %s", user)

    return user
