from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

# -------- Model --------
class SumNet(nn.Module):
    def __init__(self, input_size=5):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
    def forward(self, x):
        return self.fc(x)

# -------- Load model --------
model = SumNet(input_size=5)
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# -------- Helper function --------
def prepare_input(numbers, max_len=5):
    arr = numbers + [0] * (max_len - len(numbers))
    return torch.tensor([arr[:max_len]], dtype=torch.float32)  # truncate if >5

# -------- FastAPI App --------
app = FastAPI()

# -------- Request Schema --------
class NumbersInput(BaseModel):
    numbers: list[int]

@app.post("/predict")
def predict_sum(data: NumbersInput):
    inp = prepare_input(data.numbers)
    with torch.no_grad():
        pred = model(inp).item()
    return {
        "input": data.numbers,
        "predicted_sum": round(pred)
    }
