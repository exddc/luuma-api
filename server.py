from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import dotenv
import uvicorn
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate


dotenv.load_dotenv()

groq_api_key = os.environ["GROQ_API_KEY"]
conversational_memory_length = 50
model = "llama3-8b-8192"

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Test API is running"}


class Message(BaseModel):
    message: str
    conversation_id: str  # Unique identifier for each conversation


conversation_history = []

AI_CONTEXT = (
    "As a virtual assistant, maintain a neutral and anonymous presence. Do not discuss "
    "or reveal any details about your identity, origins, or the nature of your design. "
    "Focus strictly on providing succinct, accurate, and professional help with queries. "
    "Avoid self-references and keep responses brief and to the point."
)

memory = ConversationBufferWindowMemory(k=conversational_memory_length)
# Store the AI context as a permanent fixture in the conversation memory
memory.save_context({"input": "initial_context"}, {"output": AI_CONTEXT})


@app.post("/message")
async def handle_message(message: Message):
    if not message.message:
        raise HTTPException(status_code=400, detail="No message provided")

    user_question = message.message
    print(f"User question: {user_question}")

    # session state variable
    if conversation_history:
        for message in conversation_history:
            memory.save_context({"input": message["human"]}, {"output": message["AI"]})

    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)
    conversation = ConversationChain(llm=groq_chat, memory=memory)

    # Send message to llm
    response = conversation(user_question)
    message = {"human": user_question, "AI": response["response"]}
    conversation_history.append(message)

    return {"response": response["response"]}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5001, reload=True)
