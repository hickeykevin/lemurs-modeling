import torch
import pytest
import numpy as np
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
    def dummy_net_forward(x, uid, prev_pred=None):
        received_user_idx.append(uid.clone())
        return torch.zeros(batch_size, num_classes)
        
    dropout_module.forward = dummy_net_forward
    dropout_module.model_step(batch)
    
    assert len(received_user_idx) == 1
    assert torch.all(received_user_idx[0] == 0) # All user IDs should have dropped out to 0


def test_health_lit_module_ablation():
    """
    Tests SimpleLSTM ablation configurations:
    1. use_user_embedding=False (sequence data only)
    2. use_sequence_data=False (user embedding only)
    """
    batch_size = 4
    time_steps = 24
    input_size = 2
    num_classes = 5
    num_users = 10
    
    # -------------------------------------------------------------
    # Case 1: Sequence data only (use_user_embedding=False)
    # -------------------------------------------------------------
    net_seq_only = SimpleLSTM(
        input_size=input_size, 
        hidden_size=16, 
        num_layers=1, 
        output_size=num_classes,
        user_embedding_dim=8,
        use_user_embedding=False,
        use_sequence_data=True
    )
    # Even if init_user_embedding is called (e.g. by health module), it should not build embedding
    net_seq_only.init_user_embedding(num_users=num_users)
    assert not hasattr(net_seq_only, "user_embedding")
    assert net_seq_only.fc.in_features == 16 # Just the hidden size
    
    # Forward pass
    x = torch.randn(batch_size, time_steps, input_size)
    user_idx = torch.randint(0, num_users, (batch_size,))
    output_seq = net_seq_only(x, user_idx)
    assert output_seq.shape == (batch_size, num_classes)

    # -------------------------------------------------------------
    # Case 2: User embedding only (use_sequence_data=False)
    # -------------------------------------------------------------
    net_user_only = SimpleLSTM(
        input_size=input_size, 
        hidden_size=16, 
        num_layers=1, 
        output_size=num_classes,
        user_embedding_dim=8,
        use_user_embedding=True,
        use_sequence_data=False
    )
    # Ensure lstm is not created
    assert not hasattr(net_user_only, "lstm")
    
    # Initialize embedding
    net_user_only.init_user_embedding(num_users=num_users)
    assert hasattr(net_user_only, "user_embedding")
    assert net_user_only.fc.in_features == 8 # Just the user embedding dim
    
    # Forward pass
    output_user = net_user_only(x, user_idx)
    assert output_user.shape == (batch_size, num_classes)


def test_health_lit_module_use_prev_prediction():
    """
    Tests that the use_prev_prediction flag:
    1. Adjusts SimpleLSTM's fc projection layer size.
    2. Allows forward pass with prev_pred.
    3. Handles updates to predictions cache in LitModule.
    """
    batch_size = 2
    time_steps = 24
    input_size = 2
    num_classes = 5
    num_users = 10
    
    # 1. Instantiate network with use_prev_prediction=True
    net = SimpleLSTM(
        input_size=input_size,
        hidden_size=16,
        num_layers=1,
        output_size=num_classes,
        user_embedding_dim=8,
        use_user_embedding=True,
        use_sequence_data=True,
        use_prev_prediction=True
    )
    net.init_user_embedding(num_users=num_users)
    # in_features should be hidden_size (16) + user_embedding_dim (8) + prev_pred (1) = 25
    assert net.fc.in_features == 25
    
    # 2. Forward pass with prev_pred
    x = torch.randn(batch_size, time_steps, input_size)
    user_idx = torch.randint(0, num_users, (batch_size,))
    prev_pred = torch.randn(batch_size)
    output = net(x, user_idx=user_idx, prev_pred=prev_pred)
    assert output.shape == (batch_size, num_classes)

    # 3. Test update predictions cache logic in HealthLitModule
    def optimizer_func(params):
        return torch.optim.Adam(params, lr=0.001)

    module = HealthLitModule(net=net, optimizer=optimizer_func, use_prev_prediction=True)
    module.setup(stage="fit", num_classes=num_classes)

    # Mock datamodule/datasets for predictions_cache update
    class MockDataset:
        def __init__(self):
            self.predictions_cache = np.zeros(10, dtype=np.float32)

    class MockDataModule:
        def __init__(self):
            self.data_train = MockDataset()
            self.data_val = MockDataset()
            self.data_test = MockDataset()

    class MockTrainer:
        def __init__(self):
            self.datamodule = MockDataModule()

    # Set trainer and trigger training mode
    module.trainer = MockTrainer()
    module.train()

    # Call _update_predictions_cache directly
    idx = torch.tensor([1, 3], dtype=torch.long)
    preds = torch.tensor([2, 4], dtype=torch.float32)
    module._update_predictions_cache(idx, preds)

    # Assert cache is updated on train dataset
    assert module.trainer.datamodule.data_train.predictions_cache[1] == 2.0
    assert module.trainer.datamodule.data_train.predictions_cache[3] == 4.0


if __name__ == "__main__":
    test_health_lit_module()

