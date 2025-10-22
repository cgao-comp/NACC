import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# a = torch.rand([5,5])
# b = torch.rand([5,5])
# print('a=',a)
# print('b=',b)
#
# c = torch.mm(b, a.T)
# d = torch.mm(a, b.T)
#
# print('c=',c)
# print('d=',d)
#
# print('c.diag', c.diag())
# print('d.diag', d.diag())

a = torch.rand([3,3])
print(a)
b = torch.log(torch.diag(a))
c = torch.mean(b)
print(c)
