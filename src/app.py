import uvicorn
from backend.router import router
from fastapi import FastAPI

app = FastAPI(title="Animal Face Classifier", version="1.0.0")

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
