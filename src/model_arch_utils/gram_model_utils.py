import torch


def G_p(ob, p):
    temp = ob.detach()

    temp = temp ** p
    temp = temp.reshape(temp.shape[0], temp.shape[1], -1)
    temp = (torch.matmul(temp, temp.transpose(dim0=2, dim1=1))).sum(dim=2)
    temp = (temp.sign() * torch.abs(temp) ** (1 / p)).reshape(temp.shape[0], -1)
    return temp


def gram_record(t, is_collecting: bool):
    # For Gram ood detection
    if is_collecting:
        feature = [G_p(t, p=p).unsqueeze(0) for p in range(1, 11)]
        feature = torch.cat(feature).transpose(0, 1)  # shape=[samples,powers,feature]
        return feature
