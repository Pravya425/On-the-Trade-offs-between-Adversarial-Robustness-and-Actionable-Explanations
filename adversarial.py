import torch
import torch.nn as nn

def pgd_attack(model, x, y, epsilon=0.1, alpha=0.01, iters=10):
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)

    for _ in range(iters):
        outputs = model(x_adv)
        loss = nn.BCEWithLogitsLoss()(outputs, y)
        model.zero_grad()
        loss.backward()
        x_adv = x_adv + alpha * x_adv.grad.sign()
        eta = torch.clamp(x_adv - x, min=-epsilon, max=epsilon)
        x_adv = torch.clamp(x + eta, 0, 1).detach().requires_grad_(True)

    return x_adv
