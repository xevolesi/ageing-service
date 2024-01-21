from pydantic import BaseModel


class GenerationDAO(BaseModel):
    ages: list[int]
    images: list[bytes]
