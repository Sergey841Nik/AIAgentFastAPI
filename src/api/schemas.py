from pydantic import BaseModel

class UploadFileSchemas(BaseModel):
    filename: str
    content_type: str
    size: int