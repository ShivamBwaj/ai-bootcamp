from pydantic import BaseModel, Field
from typing import Optional

class RAGRequest(BaseModel):
    query: str=Field(..., description="The user's query for retrieval-augmented generation.")

class RAGUsedContext(BaseModel):
    image_url: str=Field(..., description="The URL of the image used to answer the question")
    price: Optional[float]=Field(..., description="The price of the item used to answer the question")
    description: str=Field(..., description="The description of the item used to answer the question")

class RAGResponse(BaseModel):
    request_id: str=Field(..., description="The request ID.")
    answer: str=Field(..., description="The retrieved information for augmentation.")
    used_context: list[RAGUsedContext]=Field(..., description="The used context.")