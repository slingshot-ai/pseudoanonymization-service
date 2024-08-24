from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

from ashley_protos.care.ashley.contracts.common.v1 import *
from ashley_protos.care.ashley.contracts.internal.v1 import *
from fastapi import FastAPI, HTTPException, Request, Response

from pseudoanonymize.ashley_protos_utils import extract_conv_from_event_log, update_message_contents
from pseudoanonymize.config import deanonymizer, get_pipeline
from pseudoanonymize.exceptions import MaxRetriesExceededException
from pseudoanonymize.models import (
    AnonymizeRequest,
    AnonymizeResponse,
    DeanonymizeRequest,
    DeanonymizeResponse,
    PseudoanonymizeResponse,
)
from pseudoanonymize.utils import chunk_by_line, flatten_replacement_dict

app = FastAPI()

anonymization_pieline = get_pipeline("GPT-4o")


@app.post("/anonymize_event_log")
async def anonymize_event_log(request: Request):
    """
    Takes a request containing an EventLog protobuf and returns an anonymized EventLog protobuf.
    """
    try:
        request_body = await request.body()
        events = EventLog.FromString(request_body)
        conv = extract_conv_from_event_log(events)
        # print(conv)

        final_anonymized_conversation = ""
        for conv_chunk in chunk_by_line(conv, 4000):
            # anonymized_conversation, replacement_dict = anonymization_pieline.predict({"text": conv_chunk})
            output = anonymization_pieline.retry_prediction({"text": conv_chunk}, 3)
            anonymized_conversation = output["anonymized_text"]
            replacement_dict = output["replacement_dict"]
            print(replacement_dict)
            final_anonymized_conversation += anonymized_conversation + "\n"

        assert len(conv.split("\n")) == len(
            final_anonymized_conversation.split("\n")
        )  # The number of lines before and after anonymization should be the same
        anonymized_event_log = update_message_contents(events, final_anonymized_conversation)
        return Response(content=anonymized_event_log.SerializeToString(), media_type="application/protobuf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize(request: AnonymizeRequest):
    try:
        # prediction = anonymizer.retry_prediction({"text": request.text}, request.retries)
        prediction = anonymization_pieline.predict({"text": request.text})
        flattened_dict = flatten_replacement_dict(prediction["replacement_dict"])
        return AnonymizeResponse(anonymized_text=prediction["anon_text"], replacement_dict=flattened_dict)
    except MaxRetriesExceededException as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/anonymize_v2", response_model=AnonymizeResponse)
async def anonymize_v2(request: AnonymizeRequest):
    try:
        text_chunks = chunk_by_line(request.text, 2000)
        print(len(text_chunks))
        anonymized_text_list = []
        replacement_dict = {}

        with ThreadPoolExecutor() as executor:
            futures = []
            for chunk in text_chunks:
                future = executor.submit(anonymization_pieline.retry_prediction, {"text": chunk}, request.retries)
                futures.append(future)

            for future in futures:
                try:
                    prediction = future.result()
                except MaxRetriesExceededException as e:
                    raise HTTPException(status_code=500, detail=str(e))

                anonymized_text_list.append(prediction["anonymized_text"])
                replacement_dict.update(prediction["replacement_dict"])

        anonymized_text = "\n".join(anonymized_text_list)
        return AnonymizeResponse(anonymized_text=anonymized_text, replacement_dict=replacement_dict)
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
        anonymized_prediction = anonymization_pieline.retry_prediction({"text": request.text}, request.retries)
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


@app.post("/anonymize_ash_conversation")
async def anonymize_ash_conversation(request: Request):
    def gather_text(entries, max_length, sep_seq):
        """
        Gather text from user and therapist messages within the given length limit.

        Args:
            entries (list): List of conversation entries.
            max_length (int): Maximum length of the gathered text.
            sep_seq (str): Separator sequence used between messages.

        Returns:
            tuple: A tuple containing gathered text, indices, and types.
        """
        gathered_text = ""
        indices = []
        types = []

        for j in range(len(entries)):
            user_msg = entries[j].user_message.content
            ther_msg = entries[j].therapist_message.content

            if user_msg and len(gathered_text) + len(user_msg) <= max_length:
                gathered_text += user_msg + sep_seq
                indices.append(j)
                types.append("user")

            if ther_msg and len(gathered_text) + len(ther_msg) <= max_length:
                gathered_text += ther_msg + sep_seq
                indices.append(j)
                types.append("therapist")

        return gathered_text, indices, types

    def update_entries_with_anonymized_text(entries, indices, types, anon_messages):
        """
        Update the conversation entries with the anonymized text.

        Args:
            entries (list): List of conversation entries.
            indices (list): Indices of the entries to be updated.
            types (list): Types of the messages (user or therapist).
            anon_messages (list): Anonymized messages.
        """
        for j_idx, msg_type in zip(indices, types):
            if msg_type == "user":
                entries[j_idx].user_message.content = anon_messages.pop(0)
            else:
                entries[j_idx].therapist_message.content = anon_messages.pop(0)

    request_body = await request.body()
    conv = ReadConversationForUserResponse.FromString(request_body)
    new_sessions = deepcopy(conv.sessions)

    max_length = 2000
    sep_seq = "-\n-"

    for i in range(len(new_sessions)):
        entries = new_sessions[i].entries
        gathered_text, indices, types = gather_text(entries, max_length, sep_seq)

        if gathered_text:
            anon_messages, _ = anonymization_pieline.predict(dict(text=gathered_text.strip()))
            anon_messages.split(sep_seq)
            update_entries_with_anonymized_text(entries, indices, types, anon_messages)

    conv.sessions[:] = new_sessions
    return Response(content=conv.SerializeToString(), media_type="application/protobuf")


# TODO: change endpoint called from the kotlin side to /anonymize_ash_conversation. This is just a temporary solution.
@app.post("/read_conversation_for_user")
async def anonymize_ash_conversation_v1(request: Request):
    return await anonymize_ash_conversation(request)
