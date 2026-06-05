import torch

# x = torch.rand(3 , requires_grad=True)

# print(x)

# y = x+2
# print(y)

# z = y*y*2
# z = z.mean()
# print(z)

# z.backward() #dz/dx
# print(x.grad)

# z = torch.tensor([1.0,2.0,3.0])

# print(z)
# print(z.mean())

# x = torch.rand(3, requires_grad=True)
# print(x)

# x.requires_grad = False
# x.detach()
#with torch.no_grad():

# x.requires_grad_(False)
# print(x)

# y = x.detach()
# print(y)

# with torch.no_grad():
#     y = x+2
#     print(y)


weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    print(model_output)
    model_output.backward(retain_graph=True)
    print(f"{weights=} and {weights.grad=}")
    weights.grad.zero_()

# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()