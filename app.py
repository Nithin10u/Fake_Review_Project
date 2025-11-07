import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import os

# -----------------------------
# ✅ Configuration
# -----------------------------
MODEL_DIR = "BERT/saved_model"


# -----------------------------
# ✅ Load Model and Tokenizer
# -----------------------------
try:
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
except Exception as e:
    raise Exception(f"Error loading model/tokenizer: {e}")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -----------------------------
# ✅ Initialize FastAPI App
# -----------------------------
app = FastAPI(
    title="Fake Review Detection API",
    description="""
    A REST API for classifying **product reviews** as Real (Genuine) or Fake (Spam/Generated)
    using a fine-tuned **BERT model**.

    ---
    **Usage:**
    - Endpoint: `/predict`
    - Method: POST
    - Body:
    ```json
    {
      "text": "Your review text here"
    }
    ```
    **Response Example:**
    ```json
    {
      "label": "Real",
      "class_id": 1,
      "confidence": 0.89,
      "probabilities": {
        "Fake": 0.11,
        "Real": 0.89
      }
    }
    ```
    """,
    version="2.0.0",
    servers=[
        {"url": "http://127.10.1.1:8000", "description": "Local Development Server"}
    ],
)

# -----------------------------
# ✅ Request Model
# -----------------------------
class ReviewText(BaseModel):
    text: str

# -----------------------------
# ✅ Prediction Endpoint
# -----------------------------
@app.post("/predict")
async def predict(review: ReviewText):
    try:
        # Tokenize input text
        inputs = tokenizer(
            review.text,
            return_tensors="pt",
            max_length=128,
            padding="max_length",
            truncation=True
        )

        # Move tensors to device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # Run inference
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence = probs.max().item()
            predicted_class = torch.argmax(probs, dim=1).item()

        # -----------------------------
        # ✅ Label mapping (adjust if needed)
        # -----------------------------
        # Ensure this matches your dataset encoding
        # Example: 0 = Fake, 1 = Real
        label_map = {0: "Fake", 1: "Real"}
        label = label_map.get(predicted_class, "Unknown")

        # Prepare response
        response = {
            "label": label,
            "class_id": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "Fake": round(probs[0][0].item(), 4),
                "Real": round(probs[0][1].item(), 4)
            }
        }

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# -----------------------------
# ✅ Root Endpoint
# -----------------------------
@app.get("/")
async def root():
    return {"message": "Fake Review Classifier API is up and running!"}
