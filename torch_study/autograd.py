import torch
x = torch.ones(2,2,requires_grad=True)
print(x)
y = x+2
print(y)  #y是计算的结果，所以它有grad_fn属性
print(y.grad_fn)
z=y*y*3
out=z.mean()
print(z,out)
z.backward(torch.ones(z.shape))
print(x.grad)


x = torch.randn(3, requires_grad=True)
y = x * 2
# L2或欧几里德范数 。
while y.data.norm() < 1000:
    y = y * 2
print(y)
v = torch.tensor([1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)