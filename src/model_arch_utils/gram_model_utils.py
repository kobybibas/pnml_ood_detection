import torch


def G_p(ob, p):
    temp = ob.detach()

    temp = temp ** p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = (torch.matmul(temp, temp.transpose(dim0=2, dim1=1))).sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)

    return temp.cpu().numpy()
