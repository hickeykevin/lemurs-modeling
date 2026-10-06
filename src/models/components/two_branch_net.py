import torch
import torch.nn as nn


class TwoBranchNet(nn.Module):
    """Sensor window and survey-item history through separate recurrent branches, fused late.

    * Sensor branch: an LSTM over the sampler's window ``x`` of shape ``[B, T, F]`` (mean-pooled).
    * Item branch: a GRU over the user's previous surveys. The history is the tail of the static
      vector (``demographics``), ``lags`` blocks of ``step_dim`` columns with lag 1 (most recent)
      first; it is reversed so the GRU reads oldest to newest. The last column of every block is the
      gap in hours to that survey (log-transformed here). Steps are normalised with a BatchNorm over
      the step features; missing surveys are zero rows with their own missing flag.
    * Head: the two embeddings plus the remaining static columns ("extras": previous labels, sleep,
      demographics) go through a small MLP.

    ``use_sensor_branch`` / ``use_item_branch`` / ``use_static_extras`` switch parts off, which gives
    the sensor-only and items-only ablations on exactly the same cohort and folds.
    ``modality_dropout`` zeroes one embedding at random per training sample (each branch with
    probability ``modality_dropout / 2``), so the model cannot lean on a single stream.
    """

    def __init__(
        self,
        input_size: int = 1,
        sensor_hidden: int = 32,
        item_hidden: int = 32,
        dropout: float = 0.2,
        head_hidden: int = 64,
        output_size: int = 2,
        use_sensor_branch: bool = True,
        use_item_branch: bool = True,
        use_static_extras: bool = True,
        modality_dropout: float = 0.0,
        demographics_dim: int = 0,
        item_lags: int = 0,
        item_step_dim: int = 0,
    ):
        super().__init__()
        if not 0.0 <= modality_dropout <= 1.0:
            raise ValueError(f"modality_dropout must be in [0, 1], got {modality_dropout}")
        if not (use_sensor_branch or use_item_branch or use_static_extras):
            raise ValueError("At least one of the sensor branch, item branch or static extras must be enabled.")
        self.sensor_hidden = sensor_hidden
        self.item_hidden = item_hidden
        self.dropout = dropout
        self.head_hidden = head_hidden
        self.output_size = output_size
        self.use_sensor_branch = use_sensor_branch
        self.use_item_branch = use_item_branch
        self.use_static_extras = use_static_extras
        self.modality_dropout = modality_dropout
        self.demographics_dim = demographics_dim
        self.item_lags = item_lags
        self.item_step_dim = item_step_dim
        self._input_size = input_size
        self._build_sensor()
        self._build_items()
        self._build_head()

    # -- construction (the datamodule reports sizes after the net is instantiated) ------------------
    def _build_sensor(self) -> None:
        self.sensor_lstm = nn.LSTM(self._input_size, self.sensor_hidden, batch_first=True) if self.use_sensor_branch else None

    def _build_items(self) -> None:
        if self.use_item_branch and self.item_lags > 0 and self.item_step_dim > 0:
            self.item_norm = nn.BatchNorm1d(self.item_step_dim)
            self.item_gru = nn.GRU(self.item_step_dim, self.item_hidden, batch_first=True)
        else:
            self.item_norm, self.item_gru = None, None

    @property
    def _extras_dim(self) -> int:
        return max(0, self.demographics_dim - self.item_lags * self.item_step_dim)

    def _build_head(self) -> None:
        in_dim = (self.sensor_hidden if self.sensor_lstm is not None else 0)
        in_dim += self.item_hidden if self.item_gru is not None else 0
        in_dim += self._extras_dim if self.use_static_extras else 0
        self.head = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(max(1, in_dim), self.head_hidden),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.head_hidden, self.output_size),
        )

    def init_input_size(self, input_size: int) -> None:
        if input_size != self._input_size:
            self._input_size = input_size
            self._build_sensor()

    # The module calls these from setup(), which Lightning runs again for the test stage. They must
    # not rebuild layers when the sizes are unchanged, or the trained weights would be replaced.
    def init_demographics(self, demographics_dim: int) -> None:
        if demographics_dim != self.demographics_dim:
            self.demographics_dim = demographics_dim
            self._build_head()

    def init_item_sequence(self, lags: int, step_dim: int) -> None:
        if (lags, step_dim) != (self.item_lags, self.item_step_dim):
            self.item_lags, self.item_step_dim = lags, step_dim
            self._build_items()
            self._build_head()

    # -- forward ------------------------------------------------------------------------------------
    def _item_sequence(self, demographics: torch.Tensor) -> torch.Tensor:
        n = self.item_lags * self.item_step_dim
        block = demographics[:, demographics.shape[1] - n :]
        seq = block.reshape(-1, self.item_lags, self.item_step_dim)
        seq = torch.cat([seq[..., :-1], torch.log1p(seq[..., -1:].clamp(min=0.0))], dim=-1)  # gap in hours
        b, l, s = seq.shape
        seq = self.item_norm(seq.reshape(b * l, s)).reshape(b, l, s)
        return seq.flip(1)  # oldest -> newest

    def forward(self, x=None, demographics=None):
        sensor = items = None
        if self.sensor_lstm is not None:
            if x is None:
                raise ValueError("The sensor branch needs the sensor window x.")
            if x.shape[-1] != self._input_size:
                raise ValueError(f"x has {x.shape[-1]} features, expected {self._input_size}; call init_input_size.")
            out, _ = self.sensor_lstm(x)
            sensor = out.mean(dim=1)
        if self.item_gru is not None:
            if demographics is None:
                raise ValueError("The item branch needs the static vector (data/prev_items=<theme>).")
            _, hn = self.item_gru(self._item_sequence(demographics))
            items = hn[-1]

        if self.training and self.modality_dropout > 0 and sensor is not None and items is not None:
            r = torch.rand(sensor.shape[0], 1, device=sensor.device)
            sensor = sensor * (r >= self.modality_dropout / 2).float()
            items = items * ((r < self.modality_dropout / 2) | (r >= self.modality_dropout)).float()

        parts = [p for p in (sensor, items) if p is not None]
        if self.use_static_extras and self._extras_dim > 0:
            if demographics is None:
                raise ValueError("static extras requested but no static vector was provided.")
            parts.append(demographics[:, : self._extras_dim])
        return self.head(torch.cat(parts, dim=-1))


if __name__ == "__main__":
    _ = TwoBranchNet()
