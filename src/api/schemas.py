from typing import Self

from pydantic import BaseModel, EmailStr, Field, ConfigDict, model_validator

from src.api.utils import hash_password


class UploadFileSchemas(BaseModel):
    filename: str
    content_type: str
    size: int


class AskResponse(BaseModel):
    response: str


class EmailModel(BaseModel):
    email: EmailStr = Field(description="Электронная почта")
    model_config = ConfigDict(from_attributes=True)


class UserAddDB(EmailModel):
    password: str = Field(description="Пароль в формате HASH-строки")

    @model_validator(mode="after")
    def hash_the_password(self) -> Self:
        self.password = hash_password(self.password)
        return self


class UserAuth(EmailModel):
    password: str = Field(
        min_length=5, max_length=50, description="Пароль, от 5 до 50 знаков"
    )


class UserInfo(EmailModel):
    pass
