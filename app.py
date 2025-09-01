import torch
import torch.nn as nn
from fastapi import FastAPI

# -------- Model --------
class SumNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
    def forward(self, x):
        return self.fc(x)

# Load model
model = SumNet(input_size=5)
model.load_state_dict(torch.load("model.pth"))
model.eval()


app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sum Prediction API is running ✅"}

@app.post("/predict/")
def predict(numbers: list[float]):
    # Pad input to length 4
    numbers = numbers + [0]*(4-len(numbers))
    x = torch.tensor([numbers], dtype=torch.float32)
    pred = model(x).item()
    return {"prediction": round(pred)}
# -------- Test with user input --------
# while True:
#     nums = input("Enter numbers separated by space (or 'q' to quit): ")
#     if nums.lower() == "q":
#         break
    
#     nums = [int(x) for x in nums.split()]
#     # Pad input to length 5
#     nums = nums + [0] * (5 - len(nums)) if len(nums) < 5 else nums[:5]

#     x = torch.tensor([nums], dtype=torch.float32)
#     pred = model(x).item()
#     print(f"🔢 Predicted Sum: {round(pred)}")
