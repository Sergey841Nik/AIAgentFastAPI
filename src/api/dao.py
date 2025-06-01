from logging import Logger

from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import select

from src.config import setup_logger
from src.core.models import Base, User


logger: Logger = setup_logger(__name__)


class AuthDao:
    model: Base = User

    def __init__(self, session: AsyncSession) -> None:
        self._session: AsyncSession = session

    async def find_one_or_none(self, filters: BaseModel) -> User | None:
        filter_dict = filters.model_dump(exclude_unset=True)
        logger.info(
            "Поиск одной записи %s по фильтрам: %s", self.model.__name__, filter_dict
        )
        try:
            query = select(self.model).filter_by(**filter_dict)
            result = await self._session.execute(query)
            record = result.scalar_one_or_none()

            logger.info(
                "Запись %s по фильтрам: %s",
                "найдена" if record else "не найдена",
                filter_dict,
            )
            return record
        except SQLAlchemyError as e:
            logger.error("Ошибка при поиске записи по фильтрам %s: %s", filter_dict, e)
            raise

    async def add(self, values: BaseModel):
        values_dict = values.model_dump(exclude_unset=True)
        logger.info(
            "Добавление записи %s с параметрами: %s", self.model.__name__, values_dict
        )
        try:
            new_instance = self.model(**values_dict)
            self._session.add(new_instance)
            logger.info("Запись %s успешно добавлена.", self.model.__name__)
            await self._session.flush()
            return new_instance
        except SQLAlchemyError as e:
            logger.error("Ошибка при добавлении записи: %s", e)
            raise
