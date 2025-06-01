from fastapi import FastAPI
import uvicorn

from src.api.views import router as router_api

app = FastAPI()

app.include_router(router_api)

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
