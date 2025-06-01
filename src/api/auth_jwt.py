
from logging import Logger

from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import EmailStr

from src.api.dao import AuthDao
from src.api.utils import validate_password, encoded_jwt
from src.api.schemas import EmailModel
from src.core.models import User
from src.core.db_helper import db_helper
from src.config import setup_logger

logger: Logger = setup_logger(__name__)


async def validate_auth_user(
    email: EmailStr,
    password: str,
    session: AsyncSession,
):
    unauthed_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid email or password",
    )

    dao = AuthDao(session=session)
    user = await dao.find_one_or_none(filters=EmailModel(email=email))

    if not user:
        raise unauthed_exc  # если пользователь не найден

    if not validate_password(
        password=password, hash_password=user.password
    ):  # проверяем пароль
        raise unauthed_exc

    return user


# создание токена
def create_jwt(
    token_type: str,
    token_data: dict,
) -> str:
    jwt_payload = {"type": token_type}
    jwt_payload.update(token_data)
    return encoded_jwt(
        payload=jwt_payload,
    )


# создание аксес токена
def create_access_token(user: User) -> str:
    jwt_payload = {
        "sub": str(user.email),
    }
    return create_jwt(
        token_type="accses",
        token_data=jwt_payload,
    )
