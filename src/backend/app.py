from fastapi import FastAPI

from .router import router

app = FastAPI(title="Animal Face Classifier", version="1.0.0")

app.include_router(router)