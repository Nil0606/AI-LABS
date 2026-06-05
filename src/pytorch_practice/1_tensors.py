"""
Author: Nil Patel
Topic: Tensors
"""

import numpy
import torch

# print(torch.empty(3,4,4,1))
# print(torch.rand(12))
# print(torch.ones(12,3,2))

# x = torch.zeros(1,4,2,dtype=torch.double)

# x = torch.tensor([12,3,2])
# y = torch.tensor([3,6,5])

# z = x+y
# z = torch.add(x,y)
# print(x+y)
# print(x.size())

# y.add_(x) # every function starts with _ do inplace operation
# print(y)

# z = x - y

# z = torch.mul(x,y)

# z = x/y
# print(z)

# x = torch.rand(5,3)
# print(x)
# print(x[1,0:6])

# print(x[1,1].item()) # only if one elements in tensor

# x = torch.rand(3,4)

# print(x)

# y = x.view(-1, 6) # to view 2 * 4 metrix as 2 * 6 

# print(y)

## numpy

# a = torch.ones(5)


# print(a)
# b = a.numpy()

# print(b)

# print(type(b))


# a.add_(1)

# print(a)
# print(b)



# a = numpy.ones(5)

# print(a)
# b = torch.from_numpy(a)

# print(f"{a=} and {b=}")

# a+=1

# print(f"{a=} and {b=}")
#both printing same
# only happen when tensor  on cpu
# numpy only handle cpu tenor not gpu tenor

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5, device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x+ y 
#     z = z.to("cpu")
# print(torch.cuda.is_available())

# x = torch.ones(5, requires_grad=True)
# print(x)