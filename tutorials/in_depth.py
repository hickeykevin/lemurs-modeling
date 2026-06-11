import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule
from omegaconf import DictConfig, OmegaConf

# Setup the project root
# This adds the project root to the python path so imports like `from src import utils` work.
# It also loads environment variables from .env.
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

warnings.filterwarnings("ignore", message=".*NVML.*")
warnings.filterwarnings("ignore", message=".*LeafSpec.*")

from src.utils import register_resolvers

# Register OmegaConf resolvers (like len resolver)
register_resolvers()


# A lightweight Dummy Trainer that mimics the PyTorch Lightning Trainer
class DummyTrainer:
    def __init__(self):
        self.datamodule = None
        self.model = None
        self.current_epoch = 0


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> None:
    print("\n" + "="*80)
    print("🎓 IN-DEPTH TUTORIAL: CHRONOLOGICAL HOOK SEQUENCE MIMICKING trainer.fit()")
    print("="*80)
    print("This walkthrough executes the manual PyTorch training and validation sequence,")
    print("calling datamodule, model, and callback hooks in the exact chronological order")
    print("they would run if passed directly to the Lightning trainer.")
    print("Debugger (pdb) breakpoints will trigger at each hook to allow close inspection.")
    print("Note: Batch-level breakpoints only trigger on the first batch of epoch 0.")
    print("="*80 + "\n")

    # ----------------------------------------------------------------------------------
    # 0. Instantiation Phase
    # ----------------------------------------------------------------------------------
    print("⚙️ [INSTANTIATION] Creating Datamodule, Model, and Callback")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    from src.utils.evaluation_callbacks import ClassificationMetricsCallback
    callback = ClassificationMetricsCallback(f1_average="macro", auroc_average="macro")

    # Associate trainer references (as Lightning does behind the scenes)
    dummy_trainer = DummyTrainer()
    dummy_trainer.datamodule = datamodule
    dummy_trainer.model = model
    model.trainer = dummy_trainer

    # Mock the model's log method to redirect logs to stdout, resolving metrics if needed
    def mock_log(name: str, value: Any, *args, **kwargs):
        if hasattr(value, "compute"):
            try:
                resolved_value = value.compute()
            except Exception:
                resolved_value = value
        else:
            resolved_value = value

        if isinstance(resolved_value, torch.Tensor):
            resolved_value = resolved_value.item()

        if isinstance(resolved_value, (int, float)):
            print(f"      📊 [LOG] {name}: {resolved_value:.4f}")
        else:
            print(f"      📊 [LOG] {name}: {resolved_value}")

    model.log = mock_log
    print(f"-> Instantiated {type(datamodule).__name__}")
    print(f"-> Instantiated {type(model).__name__}")
    print(f"-> Instantiated {type(callback).__name__}")
    print("-" * 60 + "\n")

    # ----------------------------------------------------------------------------------
    # 1. Fit Start Hooks
    # ----------------------------------------------------------------------------------
    print("\n>>> [PDB BREAKPOINT] About to call Fit Start hooks. Type 'c' or 'continue' to proceed.")
    import pdb; pdb.set_trace()

    print("🔗 [HOOK] callback.on_fit_start()")
    callback.on_fit_start(dummy_trainer, model)
    
    print("🔗 [HOOK] model.on_fit_start()")
    model.on_fit_start()
    print("-" * 60 + "\n")

    # ----------------------------------------------------------------------------------
    # 2. Data Preparation / Setup Hooks
    # ----------------------------------------------------------------------------------
    print("\n>>> [PDB BREAKPOINT] About to call Data Preparation and Setup hooks. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    print("🔗 [HOOK] datamodule.prepare_data()")
    datamodule.prepare_data()
    
    print("🔗 [HOOK] datamodule.setup(stage='fit')")
    datamodule.setup(stage="fit")
    
    print("🔗 [HOOK] model.setup(stage='fit')")
    model.setup(stage="fit")
    print("-" * 60 + "\n")

    # ----------------------------------------------------------------------------------
    # 3. Configure Optimizers Hook
    # ----------------------------------------------------------------------------------
    print("\n>>> [PDB BREAKPOINT] About to call model.configure_optimizers(). Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    print("🔗 [HOOK] model.configure_optimizers()")
    opt_config = model.configure_optimizers()
    if isinstance(opt_config, dict):
        optimizer = opt_config["optimizer"]
        scheduler = opt_config.get("lr_scheduler", {}).get("scheduler", None)
    else:
        optimizer = opt_config
        scheduler = None
    print(f"-> Configured Optimizer: {type(optimizer).__name__}")
    print("-" * 60 + "\n")

    # ----------------------------------------------------------------------------------
    # 4. Build Dataloaders
    # ----------------------------------------------------------------------------------
    print("⚙️ [DATALOADERS] Building Train and Validation loaders")
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()
    print(f"   Train dataset: {len(datamodule.data_train)} samples ({len(train_loader)} batches)")
    print(f"   Validation dataset: {len(datamodule.data_val)} samples ({len(val_loader)} batches)")
    print("-" * 60 + "\n")

    # ----------------------------------------------------------------------------------
    # 5. Sanity Check Simulation
    # ----------------------------------------------------------------------------------
    print("\n>>> [PDB BREAKPOINT] About to call Sanity Check hooks. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    print("🔍 [SANITY CHECK] Running validation check on 1 batch")
    print("🔗 [HOOK] callback.on_sanity_check_start()")
    callback.on_sanity_check_start(dummy_trainer, model)
    
    # Run a single validation step for sanity check
    model.eval()
    torch.set_grad_enabled(False)
    sanity_batch = next(iter(val_loader))
    model.on_validation_batch_start(sanity_batch, 0)
    _ = model.validation_step(sanity_batch, 0)
    
    print("🔗 [HOOK] callback.on_sanity_check_end()")
    callback.on_sanity_check_end(dummy_trainer, model)
    print("-" * 60 + "\n")

    # ----------------------------------------------------------------------------------
    # 6. Train Start Hooks
    # ----------------------------------------------------------------------------------
    print("\n>>> [PDB BREAKPOINT] About to call Train Start hooks. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    print("🔗 [HOOK] callback.on_train_start()")
    callback.on_train_start(dummy_trainer, model)
    
    print("🔗 [HOOK] model.on_train_start()")
    model.on_train_start()
    print("-" * 60 + "\n")

    # ----------------------------------------------------------------------------------
    # 7. Training Epoch/Batch Loop
    # ----------------------------------------------------------------------------------
    max_epochs = 2
    for epoch in range(max_epochs):
        dummy_trainer.current_epoch = epoch
        print(f"📅 === START OF EPOCH {epoch} ===")
        
        # --- TRAINING PHASE ---
        print("\n🏋️ [TRAINING PHASE]")
        model.train()
        torch.set_grad_enabled(True)
        
        if epoch == 0:
            print("\n>>> [PDB BREAKPOINT] About to call Train Epoch Start hooks. Type 'c' to proceed.")
            import pdb; pdb.set_trace()

        print("🔗 [HOOK] callback.on_train_epoch_start()")
        callback.on_train_epoch_start(dummy_trainer, model)
        
        print("🔗 [HOOK] model.on_train_epoch_start()")
        model.on_train_epoch_start()
        
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            is_first_batch = (epoch == 0 and batch_idx == 0)

            # A. Callback & Model Batch Start hooks
            if is_first_batch:
                print("\n>>> [PDB BREAKPOINT] About to call train batch start hooks (on_train_batch_start). Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            callback.on_train_batch_start(dummy_trainer, model, batch, batch_idx)
            model.on_train_batch_start(batch, batch_idx)
            
            # B. training_step computes forward pass & loss
            if is_first_batch:
                print("\n>>> [PDB BREAKPOINT] About to call model.training_step(). Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            loss = model.training_step(batch, batch_idx)
            
            # C. Zero gradients hook
            if is_first_batch:
                print("\n>>> [PDB BREAKPOINT] About to call model.optimizer_zero_grad(). Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            try:
                model.optimizer_zero_grad(epoch, batch_idx, optimizer)
            except TypeError:
                model.optimizer_zero_grad(epoch, batch_idx, optimizer, 0)
            
            # D. Backward gradients hook
            if is_first_batch:
                print("\n>>> [PDB BREAKPOINT] About to call model.backward() hook to execute backpropagation. Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            model.backward(loss)
            
            # E. Pre-optimizer hooks
            if is_first_batch:
                print("\n>>> [PDB BREAKPOINT] About to call before-optimizer hooks (on_before_optimizer_step). Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            callback.on_before_optimizer_step(dummy_trainer, model, optimizer)
            model.on_before_optimizer_step(optimizer)
            
            # F. Optimizer weights step
            if is_first_batch:
                print("\n>>> [PDB BREAKPOINT] About to call optimizer.step() to update weights. Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            optimizer.step()
            
            # G. Batch End hooks
            if is_first_batch:
                print("\n>>> [PDB BREAKPOINT] About to call train batch end hooks (on_train_batch_end). Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            outputs = {"loss": loss}
            model.on_train_batch_end(outputs, batch, batch_idx)
            callback.on_train_batch_end(dummy_trainer, model, outputs, batch, batch_idx)
            
            epoch_loss += loss.item()
            if (batch_idx + 1) % 15 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"   Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        avg_loss = epoch_loss / len(train_loader)
        print(f"   Average Training Loss for Epoch {epoch}: {avg_loss:.4f}")

        # --- VALIDATION PHASE ---
        print("\n🧪 [VALIDATION PHASE]")
        model.eval()
        torch.set_grad_enabled(False)
        
        if epoch == 0:
            print("\n>>> [PDB BREAKPOINT] About to call validation start & validation epoch start hooks. Type 'c' to proceed.")
            import pdb; pdb.set_trace()

        print("🔗 [HOOK] callback.on_validation_start()")
        callback.on_validation_start(dummy_trainer, model)
        print("🔗 [HOOK] model.on_validation_start()")
        model.on_validation_start()
        
        print("🔗 [HOOK] callback.on_validation_epoch_start()")
        callback.on_validation_epoch_start(dummy_trainer, model)
        print("🔗 [HOOK] model.on_validation_epoch_start()")
        model.on_validation_epoch_start()
        
        for batch_idx, batch in enumerate(val_loader):
            is_first_val_batch = (epoch == 0 and batch_idx == 0)

            # A. Callback & Model validation batch start hooks
            if is_first_val_batch:
                print("\n>>> [PDB BREAKPOINT] About to call validation batch start hooks. Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            callback.on_validation_batch_start(dummy_trainer, model, batch, batch_idx)
            model.on_validation_batch_start(batch, batch_idx)
            
            # B. validation_step
            if is_first_val_batch:
                print("\n>>> [PDB BREAKPOINT] About to call model.validation_step(). Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            outputs = model.validation_step(batch, batch_idx)
            
            # C. Batch End hooks
            if is_first_val_batch:
                print("\n>>> [PDB BREAKPOINT] About to call validation batch end hooks. Type 'c' to proceed.")
                import pdb; pdb.set_trace()
            model.on_validation_batch_end(outputs, batch, batch_idx)
            callback.on_validation_batch_end(dummy_trainer, model, outputs, batch, batch_idx)
            
        print("   -> Validation batches complete. Processing epoch end metrics:")
        
        if epoch == 0:
            print("\n>>> [PDB BREAKPOINT] About to call validation epoch end and validation end hooks. Type 'c' to proceed.")
            import pdb; pdb.set_trace()

        # Validation Epoch End Hooks
        print("🔗 [HOOK] callback.on_validation_epoch_end()")
        callback.on_validation_epoch_end(dummy_trainer, model)
        print("🔗 [HOOK] model.on_validation_epoch_end()")
        model.on_validation_epoch_end()
        
        # Validation End Hooks
        print("🔗 [HOOK] callback.on_validation_end()")
        callback.on_validation_end(dummy_trainer, model)
        print("🔗 [HOOK] model.on_validation_end()")
        model.on_validation_end()
        
        if epoch == 0:
            print("\n>>> [PDB BREAKPOINT] About to call train epoch end hooks. Type 'c' to proceed.")
            import pdb; pdb.set_trace()

        # Train Epoch End Hooks
        print("\n🔗 [HOOK] model.on_train_epoch_end()")
        model.on_train_epoch_end()
        print("🔗 [HOOK] callback.on_train_epoch_end()")
        callback.on_train_epoch_end(dummy_trainer, model)
        
        # Scheduler Step
        if scheduler:
            scheduler.step()
        print("-" * 60 + "\n")

    # ----------------------------------------------------------------------------------
    # 8. Train End Hooks
    # ----------------------------------------------------------------------------------
    print("\n>>> [PDB BREAKPOINT] About to call Train End hooks. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    print("🔗 [HOOK] model.on_train_end()")
    model.on_train_end()
    print("🔗 [HOOK] callback.on_train_end()")
    callback.on_train_end(dummy_trainer, model)
    print("-" * 60 + "\n")

    # ----------------------------------------------------------------------------------
    # 9. Fit End Hooks
    # ----------------------------------------------------------------------------------
    print("\n>>> [PDB BREAKPOINT] About to call Fit End hooks. Type 'c' to proceed.")
    import pdb; pdb.set_trace()

    print("🔗 [HOOK] model.on_fit_end()")
    model.on_fit_end()
    print("🔗 [HOOK] callback.on_fit_end()")
    callback.on_fit_end(dummy_trainer, model)
    
    print("\n" + "="*80)
    print("🎉 Chronological trainer.fit() hook walkthrough completed successfully!")
    print("="*80)


if __name__ == "__main__":
    main()
