import torch
import pytest
from src.models.health_module import HealthLitModule
from src.models.components.simple_lstm import SimpleLSTM

def test_health_lit_module():
    """
    Tests the HealthLitModule logic using dummy data to ensure 
    the forward pass and training steps work correctly.
    """
    batch_size = 4
    time_steps = 24
    input_size = 2
    num_classes = 5
    
    # 1. Instantiate the underlying network
    net = SimpleLSTM(
        input_size=input_size, 
        hidden_size=16, 
        num_layers=1, 
        output_size=num_classes
    )
    
    # 2. Instantiate the LightningModule
    # We pass a simple optimizer function for the test
    def optimizer_func(params):
        return torch.optim.Adam(params, lr=0.001)
        
    module = HealthLitModule(net=net, optimizer=optimizer_func)
    module.setup(stage="fit", num_classes=num_classes)
    
    # 3. Create a dummy batch
    # x: [Batch, Time, Features]
    x = torch.randn(batch_size, time_steps, input_size)
    # y: [Batch] (Target labels 0-4)
    y = torch.randint(0, num_classes, (batch_size,))
    batch = (x, y)
    
    # 4. Test forward pass
    output = module(x)
    assert output.shape == (batch_size, num_classes)
    
    # 5. Test training step
    loss = module.training_step(batch, 0)
    assert loss is not None
    assert loss.shape == torch.Size([]) # Must be a scalar tensor
    
    # 6. Test validation / accuracy metrics
    module.validation_step(batch, 0)
    val_loss = module.val_loss.compute()
    val_acc = module.val_acc.compute()
    
    assert val_loss >= 0
    assert 0 <= val_acc <= 1.0

if __name__ == "__main__":
    test_health_lit_module()
