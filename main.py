import os
from typing import Any, Dict

import humps
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from pseudoanonymize.anonymization import Anonymizer
from pseudoanonymize.deanonymization import Deanonymizer
from pseudoanonymize.exceptions import MaxRetriesExceededException
from pseudoanonymize.utils import flatten_replacement_dict

app = FastAPI()

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

anonymizer = Anonymizer(client=client)
deanonymizer = Deanonymizer(client=client)


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


@app.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize(request: AnonymizeRequest):
    try:
        prediction = anonymizer.retry_prediction({"text": request.text}, request.retries)
        flattened_dict = flatten_replacement_dict(prediction["replacement_dict"])
        return AnonymizeResponse(anonymized_text=prediction["anonymized_text"], replacement_dict=flattened_dict)
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deanonymize", response_model=DeanonymizeResponse)
async def deanonymize(request: DeanonymizeRequest):
    try:
        prediction = deanonymizer.retry_prediction(
            {"text": request.text, "replacement_dict": request.replacement_dict}, request.retries
        )
        return DeanonymizeResponse(deanonymized_text=prediction["deanonymized_text"])
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/pseudoanonymize", response_model=PseudoanonymizeResponse)
async def pseudoanonymize(request: AnonymizeRequest):
    try:
        anonymized_prediction = anonymizer.retry_prediction({"text": request.text}, request.retries)
        anonymized_text = anonymized_prediction["anonymized_text"]
        replacement_dict = flatten_replacement_dict(anonymized_prediction["replacement_dict"])

        deanon_prediction = deanonymizer.retry_prediction(
            {"text": anonymized_text, "replacement_dict": replacement_dict}, request.retries
        )
        return PseudoanonymizeResponse(
            anonymized_text=anonymized_text,
            replacement_dict=replacement_dict,
            deanonymized_text=deanon_prediction["deanonymized_text"],
        )
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))
