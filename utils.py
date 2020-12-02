import torch
import torch.nn.functional as F



def get_softmax_out(model, loader, device, is_dac=False):
    softmax_out = []
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            if is_dac:
                softmax_out.append(F.softmax(model(data)[:,:-1], dim=1))
            else:
                softmax_out.append(torch.exp(model(data)))
    return torch.cat(softmax_out).cpu().numpy()
