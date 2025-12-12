import torch
from inverse_model import MIGInverseModel
from mig_dataset import MIGInverseDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------
# 1. BUILD THE MODEL WITH THE EXACT TRAINING ARCHITECTURE
# -----------------------------------------------------------
model = MIGInverseModel(
    Nx=32, Ny=32, Nz=16,
    Hs=16, Ws=16,
    latent_channels=20,   # ← MUST MATCH TRAIN FILE
    fno_width=36,         # ← MUST MATCH TRAIN FILE
    fno_layers=4,
    modes_x=12, modes_y=12, modes_z=8
).to(device)

# -----------------------------------------------------------
# 2. LOAD THE TRAINED WEIGHTS
# -----------------------------------------------------------
state = torch.load("mig_inverse_fno.pt", map_location=device)
model.load_state_dict(state)
model.eval()
print("✓ model loaded successfully")

# -----------------------------------------------------------
# 3. CREATE ONE SYNTHETIC TEST SAMPLE
# -----------------------------------------------------------
test_set = MIGInverseDataset(
    n_samples=1,
    Nx=32, Ny=32, Nz=16,
    dx=1e-3, dy=1e-3, dz=1e-3,
    Nsx=16, Nsy=16,
    sensor_offset=0.01,
    noise_sigma=0.05,
    device=device
)

sample = test_set[0]
B = sample["B"].unsqueeze(0).to(device)
J_true = sample["J"].unsqueeze(0).to(device)

# -----------------------------------------------------------
# 4. RUN INFERENCE
# -----------------------------------------------------------
with torch.no_grad():
    J_pred = model(B, test_set.X, test_set.Y, test_set.Z)

# -----------------------------------------------------------
# 5. REPORT RECONSTRUCTION ERROR
# -----------------------------------------------------------
mse = torch.nn.functional.mse_loss(J_pred, J_true).item()
print("reconstruction error:", mse)
