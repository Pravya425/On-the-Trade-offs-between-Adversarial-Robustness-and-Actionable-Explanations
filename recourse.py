import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

# ------------------------
# SCFE: Sparse CF via L1
# ------------------------
def generate_scfe(model, x_orig, target=1, max_iter=1000, step_size=0.01, lambda_reg=0.05):
    """
    Perform a gradient-based sparse recourse.
    """
    model.eval()
    x = torch.tensor(x_orig, dtype=torch.float32, requires_grad=True)
    target_tensor = torch.tensor([[target]], dtype=torch.float32)

    optimizer = torch.optim.Adam([x], lr=step_size)

    for i in range(max_iter):
        pred = model(x)
        loss = torch.nn.BCELoss()(pred, target_tensor) + lambda_reg * torch.norm(x - x_orig, p=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if torch.round(pred).item() == target:
            break

    x_cf = x.detach().numpy()
    return x_cf


# ------------------------
# C-CHVAE: Prototype-based
# ------------------------
def generate_cchvae(model, x_orig, data_latent, target=1, k=5):
    """
    Find nearest valid example of target class.
    """
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(data_latent, dtype=torch.float32)).numpy().flatten()

    target_indices = np.where(np.round(preds) == target)[0]
    if len(target_indices) == 0:
        return None

    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(data_latent[target_indices])
    _, indices = neigh.kneighbors([x_orig])

    x_cf = data_latent[target_indices[indices[0][0]]]
    return x_cf
