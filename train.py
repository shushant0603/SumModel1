import torch
import torch.nn as nn
import torch.optim as optim

# -------- Training Data (variable length with padding = 0) --------
X_train = [
    [1, 2, 0, 0, 0],
    [3, 4, 0, 0, 0],
    [6, 7, 8, 0, 0],
    [10, 11, 12, 0, 0],
    [5, 7, 2, 0, 0],
    [1, 2, 9, 0, 0],
    [1, 2, 3, 6, 9]
]
y_train = [3, 7, 21, 33, 14, 12, 21]  # sums

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# -------- Model --------
class SumNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
    def forward(self, x):
        return self.fc(x)

model = SumNet(input_size=5)  # max length = 5
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# -------- Train --------
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# -------- Save Model --------
torch.save(model.state_dict(), "model.pth")
print("✅ Model trained and saved as model.pth")
