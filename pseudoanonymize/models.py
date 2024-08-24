from typing import Any, Dict

import humps
from pydantic import BaseModel


class CamelModel(BaseModel):
    id: str | None = None

    class Config:
        alias_generator = humps.camelize
        populate_by_name = True


class AnonymizeRequest(CamelModel):
    text: str
    retries: int = 5


class DeanonymizeRequest(CamelModel):
    text: str
    replacement_dict: Dict[Any, str]
    retries: int = 5


class AnonymizeResponse(CamelModel):
    anonymized_text: str
    replacement_dict: Dict[str, str]


class DeanonymizeResponse(CamelModel):
    deanonymized_text: str


class PseudoanonymizeResponse(CamelModel):
    anonymized_text: str
    replacement_dict: Dict[str, str]
    deanonymized_text: str
