from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="Federated Meta-Learning AI Assistant API"
)


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {
        "message": "AI Assistant Backend is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat")
def chat(request: ChatRequest):

    response = (
        "Hello! I received your message. "
        "The trained AI model will be connected here later."
    )

    return {
        "user_message": request.message,
        "response": response
    }