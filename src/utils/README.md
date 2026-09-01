# General Utilities (`src/utils/`)

This directory contains shared infrastructure and system utilities:

- **[pylogger.py](pylogger.py)**: Ranked multi-GPU / process logger.
- **[rich_utils.py](rich_utils.py)**: Rich formatting for configuration printing, tags enforcement, and exception trees.
- **[instantiators.py](instantiators.py)**: Dynamic instantiation of Hydra callbacks and loggers.
- **[logging_utils.py](logging_utils.py)**: WandB, CSV, and TensorBoard hyperparameter logging helpers.
- **[resolvers.py](resolvers.py)**: Custom OmegaConf string and math interpolation resolvers.
- **[database_service.py](database_service.py)**: Database connector and query interface for raw participant sensing records.
- **[checkpoint_compat.py](checkpoint_compat.py)**: PyTorch 2.6+ checkpoint loading compatibility layer.
- **[utils.py](utils.py)**: Task wrappers, metric retrieval, and general helpers.
