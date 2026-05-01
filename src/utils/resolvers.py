from omegaconf import OmegaConf, ListConfig, DictConfig


def register_resolvers() -> None:
    """Registers all custom OmegaConf resolvers used across the project.

    Call this once at the entry point (e.g. ``src/train.py``) before Hydra
    composes the config.

    Resolvers registered:
        - ``len``: Returns the length of a list-valued config node.
          Example usage in YAML::

              input_size: ${len:${data.modalities}}

          Note: Use ``${len:${data.modalities}}`` (nested interpolation) so
          that OmegaConf resolves the list first and passes the actual
          ``ListConfig`` object to this resolver — not the raw key string.
    """
    def _get_len(x) -> int:
        """Returns len() of a list-like value.

        Args:
            x: A ``ListConfig``, list, tuple, or dict resolved by OmegaConf.

        Returns:
            int: Number of elements.
        """
        if isinstance(x, (list, tuple, ListConfig, DictConfig, dict)):
            return len(x)
        if isinstance(x, str):
            return len(x)
        try:
            return len(x)
        except TypeError:
            return 0

    OmegaConf.register_new_resolver("len", _get_len, replace=True)
