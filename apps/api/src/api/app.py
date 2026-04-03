from fastapi import FastAPI
from api.core.config import config
from api.api.middleware import RequestIDMiddleware
import logging
from fastapi.middleware.cors import CORSMiddleware
from api.api.endpoints import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger=logging.getLogger(__name__)

app=FastAPI()
app.add_middleware(RequestIDMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(api_router)

