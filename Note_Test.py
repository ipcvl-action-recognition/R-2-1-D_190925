import torch
y_pred_index = [[1, 0, 0, 1, 0, 1, 0, 1, 0, 0]]
y = [1, 0, 0, 0, 0, 0, 1, 1, 1, 0]

y_pred_index = torch.FloatTensor(y_pred_index)
y = torch.FloatTensor(y)
y_pred_index = y_pred_index.int()
y = y.int()

print("y_pred_index = ", y_pred_index.shape)
print("y = ", y.shape)

print("y_pred_index = ", y_pred_index)
print("y = ", y)

print("correct = ", (y_pred_index == y).sum().item())