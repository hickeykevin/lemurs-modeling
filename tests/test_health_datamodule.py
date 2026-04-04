import torch
import pytest
import rootutils
from pathlib import Path

rootutils.setup_root(Path(__file__).parent, indicator=".project-root", pythonpath=True)

from src.data.health_datamodule import HealthDataModule

@pytest.mark.parametrize("modalities, modality_cols", [
    (["step"], {"step": "steps"}),
    (["step", "calorie"], {"step": "steps", "calorie": "calories"})
])
def test_health_datamodule(modalities, modality_cols) -> None:
    """
    Tests the HealthDataModule to verify it can fetch data for multiple modalities,
    split it, and produce correctly shaped tensors.
    """
    dm = HealthDataModule(
        modalities=modalities, 
        modality_cols=modality_cols, 
        batch_size=4
    )
    
    # 1. Setup
    dm.setup()
    
    # 2. Verify dataloaders
    train_loader = dm.train_dataloader()
    assert train_loader
    
    # 3. Check batch content
    batch = next(iter(train_loader))
    x, y = batch
    
    # x shape: [Batch, Time=24, Features=N_modalities]
    assert x.ndim == 3
    assert x.shape[0] == 4
    assert x.shape[1] == 24
    assert x.shape[2] == len(modalities) # This is the generalization part!
    
    assert x.dtype == torch.float32
    assert y.dtype == torch.long

if __name__ == "__main__":
    # Test with multiple modalities
    test_health_datamodule(["step", "calorie"], {"step": "steps", "calorie": "calories"})
