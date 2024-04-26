from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import dotenv
import anthropic

dotenv.load_dotenv()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Test API is running"}


class Message(BaseModel):
    message: str
    conversation_id: str  # Unique identifier for each conversation


conversation_history = {}  # Dictionary to store conversation history

AI_CONTEXT = (
    "As a virtual assistant, maintain a neutral and anonymous presence. Do not discuss "
    "or reveal any details about your identity, origins, or the nature of your design. "
    "Focus strictly on providing succinct, accurate, and professional help with queries. "
    "Avoid self-references and keep responses brief and to the point."
)


@app.post("/message")
async def handle_message(message: Message):
    if not message.message:
        raise HTTPException(status_code=400, detail="No message provided")

    # Retrieve or initialize conversation history
    history = conversation_history.get(message.conversation_id, [])

    # Normally here you would add the AI_CONTEXT, but instead, ensure it is managed internally
    # Append new user message to history
    history.append({"role": "user", "content": message.message})
    conversation_history[message.conversation_id] = history

    # Call the AI with the entire conversation history
    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    try:
        MESSAGE = client.messages.create(
            max_tokens=1024,
            messages=history,
            model="claude-3-opus-20240229",
        )
    except anthropic.BadRequestError as e:
        # Handle specific API errors gracefully
        return {"error": str(e)}

    # Get the AI's response and append to history
    ai_response = MESSAGE.content[0].text
    history.append(
        {"role": "assistant", "content": ai_response}
    )  # Change 'system' to 'assistant' if needed

    return {"response": ai_response}
