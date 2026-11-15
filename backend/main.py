from pydantic import BaseModel, Field
from personalization.service import personalize
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
class FeedbackItem(BaseModel):
    features: list[float] = Field(min_length=8, max_length=8)
    liked: int = Field(ge=0, le=1)


class PersonalizeRequest(BaseModel):
    feedback: list[FeedbackItem]
    candidates: list[list[float]]


@app.post("/personalize")
def personalize_user(request: PersonalizeRequest):
    scores = personalize(
        [item.model_dump() for item in request.feedback],
        request.candidates,
    )
    return {"personalization_scores": scores}