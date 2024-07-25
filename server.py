"""Basic server function for the luuma api."""

import os
import json
from collections import defaultdict
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import dotenv
import uvicorn
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import logger

dotenv.load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
conversational_memory_length = 50
model = "llama-3.1-70b-versatile"

app = FastAPI()

# Initialize logger
logger = logger.get_module_logger(__name__)

# Initialize token log file and path
token_log_path = "logs/token_counts.json"
if not os.path.exists("logs"):
    os.makedirs("logs")
if not os.path.isfile(token_log_path):
    with open(token_log_path, "w") as token_file:
        json.dump(
            {"total_input_tokens": 0, "total_output_tokens": 0, "ip_usage": {}},
            token_file,
        )

# Maps each conversation_id to its specific ConversationBufferWindowMemory instance
conversation_memories = defaultdict(
    lambda: ConversationBufferWindowMemory(k=conversational_memory_length)
)

# Maps each conversation_id to its specific conversation history
conversation_histories = defaultdict(list)

AI_CONTEXT = (
    "As a virtual assistant, maintain a neutral and anonymous presence. Do not discuss "
    "or reveal any details about your identity, origins, or the nature of your design. "
    "Focus strictly on providing succinct, accurate, and professional help with queries. "
    "Avoid self-references and keep responses brief and to the point."
)


@app.get("/")
async def root():
    return {"message": "Test API is running"}


class Message(BaseModel):
    message: str
    conversation_id: str  # Unique identifier for each conversation


@app.post("/message")
async def handle_message(message: Message, request: Request):
    if not message.message:
        raise HTTPException(status_code=400, detail="No message provided")

    client_ip = request.client.host
    logger.info(f"Message from {client_ip}: {message.message}")

    # Retrieve or initialize memory for the specific conversation_id
    memory = conversation_memories[message.conversation_id]

    # Check if we need to add the initial context for new conversation
    if not conversation_histories[message.conversation_id]:
        memory.save_context({"input": "initial_context"}, {"output": AI_CONTEXT})

    user_question = message.message
    input_tokens = len(user_question.split())

    # Load past conversation context into memory
    for msg in conversation_histories[message.conversation_id]:
        memory.save_context({"input": msg["human"]}, {"output": msg["AI"]})

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    conversation = ConversationChain(llm=groq_chat, memory=memory)

    # Send message to LLM
    response = conversation(user_question)
    output_tokens = len(response["response"].split())

    message_record = {"human": user_question, "AI": response["response"]}
    conversation_histories[message.conversation_id].append(message_record)

    # Log the token data
    with open(token_log_path, "r+") as f:
        token_data = json.load(f)
        token_data["total_input_tokens"] += input_tokens
        token_data["total_output_tokens"] += output_tokens
        if client_ip in token_data["ip_usage"]:
            token_data["ip_usage"][client_ip]["input_tokens"] += input_tokens
            token_data["ip_usage"][client_ip]["output_tokens"] += output_tokens
        else:
            token_data["ip_usage"][client_ip] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
        f.seek(0)
        json.dump(token_data, f, indent=4)
        f.truncate()

    logger.info(f"Response to {client_ip}: {response['response']}")

    return {"response": response["response"]}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5001, reload=True)
