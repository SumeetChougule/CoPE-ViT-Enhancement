import torch
import numpy as np
from ViT_arc import ViT, data
from src import engine, utils

# MPS on my MacBook Pro couldn't handle the batch size in memory but it worked with cpu. So, I've set device = 'cpu'. If you've Nvidia GPU then uncomment the following line.
# device = (
#     "mps"
#     if torch.backends.mps.is_available()
#     else "cuda" if torch.cuda.is_available() else "cpu"
# )

device = "cpu"


cope_vit = ViT.ViT_Base(num_classes=3).to(device)

# Setup the optimizer to optimize our ViT model parameters using hyperparameters from the ViT paper
optimizer = torch.optim.Adam(
    params=cope_vit.parameters(),
    lr=3e-3,  # Base LR from Table 3 for ViT-* ImageNet-1k
    betas=(0.9, 0.999),
    weight_decay=0.3,
)

# setup loss fun
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model and save the training results to a dictionary
results = engine.train(
    model=cope_vit,
    train_dataloader=data.train_dataloader,
    test_dataloader=data.test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=10,
    device=device,
)

# Plot our ViT model's loss curves
utils.plot_loss_curves(results)
utils.save_model(model=cope_vit, target_dir="models", model_name="CoPE_ViT_trained.pth")
