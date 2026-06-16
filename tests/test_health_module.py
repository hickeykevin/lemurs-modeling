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
    # user_idx: [Batch]
    user_idx = torch.zeros(batch_size, dtype=torch.long)
    batch = (x, y, user_idx)
    
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


def test_health_lit_module_with_user_embeddings():
    """
    Tests HealthLitModule with dynamic user embeddings and user ID dropout.
    """
    batch_size = 4
    time_steps = 24
    input_size = 2
    num_classes = 5
    num_users = 10
    
    # 1. Instantiate network and initialize user embedding
    net = SimpleLSTM(
        input_size=input_size, 
        hidden_size=16, 
        num_layers=1, 
        output_size=num_classes,
        user_embedding_dim=8
    )
    net.init_user_embedding(num_users=num_users)
    assert hasattr(net, "user_embedding")
    assert net.fc.in_features == 16 + 8
    
    # 2. Instantiate LitModule
    def optimizer_func(params):
        return torch.optim.Adam(params, lr=0.001)
        
    module = HealthLitModule(net=net, optimizer=optimizer_func, user_id_dropout=0.5)
    module.setup(stage="fit", num_classes=num_classes)
    
    # 3. Create dummy batch with user indices
    x = torch.randn(batch_size, time_steps, input_size)
    y = torch.randint(0, num_classes, (batch_size,))
    user_idx = torch.randint(0, num_users, (batch_size,))
    batch = (x, y, user_idx)
    
    # 4. Test forward pass with user embeddings
    output = module.forward(x, user_idx)
    assert output.shape == (batch_size, num_classes)
    
    # 5. Test training step with dropout
    loss = module.training_step(batch, 0)
    assert loss is not None
    assert loss.shape == torch.Size([])
    
    # 6. Test dropout=1.0 forces mapping to 0
    dropout_module = HealthLitModule(net=net, optimizer=optimizer_func, user_id_dropout=1.0)
    dropout_module.train() # Set to train mode for dropout to activate
    
    # Override network forward to inspect user_idx
    received_user_idx = []
    def dummy_net_forward(x, uid):
        received_user_idx.append(uid.clone())
        return torch.zeros(batch_size, num_classes)
        
    dropout_module.forward = dummy_net_forward
    dropout_module.model_step(batch)
    
    assert len(received_user_idx) == 1
    assert torch.all(received_user_idx[0] == 0) # All user IDs should have dropped out to 0


if __name__ == "__main__":
    test_health_lit_module()
