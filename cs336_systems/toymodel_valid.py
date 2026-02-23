from torch import nn
import torch
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x
device = "cuda"
dtype = torch.bfloat16
in_features = 20
out_features = 5
batch_size = 8
model = ToyModel(in_features, out_features).to(device)
x = torch.rand(batch_size, in_features, device=device)
target = torch.rand(batch_size, out_features, device=device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
intermediate_dtypes = {}
def save_dtype_hook(name):
    """一个简单的 hook 函数，用于记录模块输出的 dtype"""
    def hook(model, input, output):
        intermediate_dtypes[name] = output.dtype
    return hook
model.relu.register_forward_hook(save_dtype_hook("output_of_fc1_relu"))
model.ln.register_forward_hook(save_dtype_hook("output_of_ln"))
print(f"--- 正在使用 torch.autocast(dtype={dtype}) 运行 ---")
# 清除任何可能存在的旧梯度
optimizer.zero_grad()
with torch.autocast(device_type=device, dtype=dtype):
    # 问题1: autocast 上下文中的模型参数
    param_dtype_inside = model.fc1.weight.dtype
    y_pred = model(x)  # Logits
    # 问题4: 模型的预测 logits (y_pred)
    logits_dtype = y_pred.dtype
    # 问题5: 损失 criterion(y_pred, target) -> (FP16, FP32)
    # y_pred 会被提升到 FP32 来计算损失
    loss = criterion(y_pred, target)
    loss_dtype = loss.dtype
# 问题2: fc1/relu 的输出 (从 hook 获取)
fc1_output_dtype = intermediate_dtypes.get("output_of_fc1_relu")
# 问题3: layer norm 的输出 (从 hook 获取)
ln_output_dtype = intermediate_dtypes.get("output_of_ln")
print("--- 正在运行 loss.backward() ---")
loss.backward()
# 问题6: 模型的梯度
grad_dtype = model.fc1.weight.grad.dtype
print("\n--- 结果分析 ---")
print(f"1. 模型参数 (fc1.weight) 在 autocast 上下文中: {param_dtype_inside}")
print(f"2. 第一个 fc1/relu 块的输出: {fc1_output_dtype}")
print(f"3. LayerNorm (ln) 的输出: {ln_output_dtype}")
print(f"4. 模型的预测 logits: {logits_dtype}")
print(f"5. 损失 (loss): {loss_dtype}")
print(f"6. 模型的梯度 (fc1.weight.grad): {grad_dtype}")