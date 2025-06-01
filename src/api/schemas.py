from pydantic import BaseModel, EmailStr, Field, ConfigDict

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
    password: bytes = Field(description="Пароль в формате HASH-строки")

class UserAuth(EmailModel):
    password: str = Field(
        min_length=5, max_length=50, description="Пароль, от 5 до 50 знаков"
    )
