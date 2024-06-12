from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from pseudoanonymize.anonymization import Anonymizer
from pseudoanonymize.deanonymization import Deanonymizer
from pseudoanonymize.exceptions import UnparsableLLMOutputException, MaxRetriesExceededException
from openai import OpenAI
from dotenv import load_dotenv
import os

app = FastAPI()

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)


anonymizer = Anonymizer(client=client)
deanonymizer = Deanonymizer(client=client)

class AnonymizeRequest(BaseModel):
    text: str
    retries: int = 5

class DeanonymizeRequest(BaseModel):
    text: str
    replacement_dict: Dict[Any, str]
    retries: int = 5

def flatten_replacement_dict(replacement_dict):
    flat_dict = {}
    for key, value in replacement_dict.items():
        if isinstance(key, tuple):
            for sub_key in key:
                flat_dict[sub_key] = value
        else:
            flat_dict[key] = value
    return flat_dict

@app.post("/anonymize")
async def anonymize(request: AnonymizeRequest):
    try:
        prediction = anonymizer.retry_prediction({"text": request.text}, request.retries)
        flattened_dict = flatten_replacement_dict(prediction["replacement_dict"])
        return {
            "anonymized_text": prediction["anonymized_text"],
            "replacement_dict": flattened_dict
        }
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/deanonymize")
async def deanonymize(request: DeanonymizeRequest):
    try:
        prediction = deanonymizer.retry_prediction({"text": request.text, "replacement_dict": request.replacement_dict}, request.retries)
        return {"deanonymized_text": prediction["deanonymized_text"]}
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pseudoanonymize")
async def pseudoanonymize(request: AnonymizeRequest):
    try:
        anonymized_prediction = anonymizer.retry_prediction({"text": request.text}, request.retries)
        anonymized_text = anonymized_prediction["anonymized_text"]
        replacement_dict = flatten_replacement_dict(anonymized_prediction["replacement_dict"])

        deanon_prediction = deanonymizer.retry_prediction({"text": anonymized_text, "replacement_dict": replacement_dict}, request.retries)
        return {
            "anonymized_text": anonymized_text,
            "replacement_dict": replacement_dict,
            "deanonymized_text": deanon_prediction["deanonymized_text"]
        }
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))
