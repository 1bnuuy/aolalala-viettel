from fastapi import FastAPI
from scripts.chat import router

app = FastAPI()
app.include_router(router)


@app.get("/")
def root():
    return {"status": "running"}
