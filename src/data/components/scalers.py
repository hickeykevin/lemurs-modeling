from typing import Any, Dict, List, Optional
import numpy as np
from copy import deepcopy

class SubjectScaler:
    """Scaler wrapper that applies standardizing/scaling on a per-subject level.
    
    This class wraps an arbitrary scikit-learn scaler and maintains a separate 
    fitted instance for each unique subject/user ID. If a user is not seen during 
    training, a scaler copy is dynamically fit on their data on the fly.
    """
    def __init__(self, base_scaler: Any, scale_dim: Optional[int] = None):
        self.base_scaler = base_scaler
        self.scalers: Dict[Any, Any] = {}
        self.global_scaler: Optional[Any] = None
        self.scale_dim = scale_dim

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SubjectScaler":
        """Standard fit method for compatibility, fitting a single global scaler."""
        self.global_scaler = deepcopy(self.base_scaler)
        if self.scale_dim is not None:
            self.global_scaler.fit(X[:, :self.scale_dim], y)
        else:
            self.global_scaler.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standard transform method for compatibility, applying the global scaler."""
        if self.global_scaler is not None:
            if self.scale_dim is not None:
                out = X.copy()
                out[:, :self.scale_dim] = self.global_scaler.transform(X[:, :self.scale_dim])
                return out
            else:
                return self.global_scaler.transform(X)
        return X

    def fit_by_subject(self, sequences: List[np.ndarray], user_ids: np.ndarray) -> None:
        """Fits a separate copy of base_scaler for each unique subject/user ID.
        
        Args:
            sequences (List[np.ndarray]): List of arrays of shape [Time, Features].
            user_ids (np.ndarray): Array of subject IDs corresponding to each sequence.
        """
        # Fit global scaler as a fallback
        self.global_scaler = deepcopy(self.base_scaler)
        stacked_all = np.concatenate(sequences, axis=0)  # [N*T, F]
        if self.scale_dim is not None:
            self.global_scaler.fit(stacked_all[:, :self.scale_dim])
        else:
            self.global_scaler.fit(stacked_all)
        
        # Fit per-subject scalers
        unique_users = np.unique(user_ids)
        for uid in unique_users:
            user_seqs = [sequences[i] for i, u in enumerate(user_ids) if u == uid]
            user_stacked = np.concatenate(user_seqs, axis=0)  # [N_u * T, F]
            
            scaler = deepcopy(self.base_scaler)
            if self.scale_dim is not None:
                scaler.fit(user_stacked[:, :self.scale_dim])
            else:
                scaler.fit(user_stacked)
            self.scalers[uid] = scaler

    def transform_by_subject(self, seqs_np: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
        """Transforms the sequences on a per-subject level.
        
        If a user has not been seen in the training data, a scaler copy is dynamically
        fit on the fly using their validation/test sequences.
        
        Args:
            seqs_np (np.ndarray): Multi-dimensional array of shape [N, Time, Features].
            user_ids (np.ndarray): Array of subject IDs corresponding to each sequence.
            
        Returns:
            np.ndarray: Transformed array of shape [N, Time, Features].
        """
        n, t, f = seqs_np.shape
        out = seqs_np.copy()
        
        unique_users = np.unique(user_ids)
        for uid in unique_users:
            user_mask = (user_ids == uid)
            
            # If user has not been seen before, fit standardizer dynamically on the fly
            if uid not in self.scalers:
                user_seqs = seqs_np[user_mask]  # [N_u, T, F]
                if self.scale_dim is not None:
                    user_stacked = user_seqs[:, :, :self.scale_dim].reshape(-1, self.scale_dim)
                else:
                    user_stacked = user_seqs.reshape(-1, f)
                
                scaler = deepcopy(self.base_scaler)
                scaler.fit(user_stacked)
                self.scalers[uid] = scaler
                
            # Retrieve the subject-specific scaler and transform
            scaler = self.scalers[uid]
            user_seqs = seqs_np[user_mask]  # [N_u, T, F]
            u_n, u_t, u_f = user_seqs.shape
            
            if self.scale_dim is not None:
                scaled_part = scaler.transform(user_seqs[:, :, :self.scale_dim].reshape(-1, self.scale_dim)).reshape(u_n, u_t, self.scale_dim)
                out[user_mask, :, :self.scale_dim] = scaled_part
            else:
                transformed = scaler.transform(user_seqs.reshape(-1, f)).reshape(u_n, u_t, u_f)
                out[user_mask] = transformed
            
        return out.astype(np.float32)


class DualScaler:
    """Scaler that outputs both globally standardized and subject-standardized features.
    
    For an input array of shape [N, Time, Features], it transforms the input using
    both a global scaler and a subject-specific scaler, and concatenates the results
    along the feature dimension, resulting in [N, Time, 2 * scale_dim + (Features - scale_dim)].
    """
    def __init__(self, base_scaler_global: Any, base_scaler_subject: Any, scale_dim: Optional[int] = None):
        self.global_scaler = base_scaler_global
        self.subject_scaler = SubjectScaler(base_scaler_subject, scale_dim=scale_dim)
        self.scale_dim = scale_dim

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "DualScaler":
        """Standard fit method for compatibility, fitting the global scaler."""
        if self.scale_dim is not None:
            self.global_scaler.fit(X[:, :self.scale_dim], y)
        else:
            self.global_scaler.fit(X, y)
        return self

    def fit_by_subject(self, sequences: List[np.ndarray], user_ids: np.ndarray) -> None:
        """Fits the global scaler and subject-specific scaler."""
        # Fit global scaler
        stacked_all = np.concatenate(sequences, axis=0)  # [N*T, F]
        if self.scale_dim is not None:
            self.global_scaler.fit(stacked_all[:, :self.scale_dim])
        else:
            self.global_scaler.fit(stacked_all)
            
        # Fit subject scaler
        self.subject_scaler.fit_by_subject(sequences, user_ids)

    def transform_by_subject(self, seqs_np: np.ndarray, user_ids: np.ndarray) -> np.ndarray:
        """Transforms the sequences, concatenating global and subject scaling."""
        n, t, f = seqs_np.shape
        
        # Transform globally
        if self.scale_dim is not None:
            global_scaled = (
                self.global_scaler.transform(seqs_np[:, :, :self.scale_dim].reshape(-1, self.scale_dim))
                .reshape(n, t, self.scale_dim)
                .astype(np.float32)
            )
        else:
            global_scaled = (
                self.global_scaler.transform(seqs_np.reshape(-1, f))
                .reshape(n, t, f)
                .astype(np.float32)
            )
            
        # Transform by subject
        subject_transformed = self.subject_scaler.transform_by_subject(seqs_np, user_ids)
        
        # Concatenate global_scaled and subject_transformed along feature dimension
        if self.scale_dim is not None:
            subject_scaled_part = subject_transformed[:, :, :self.scale_dim]
            untouched_part = subject_transformed[:, :, self.scale_dim:]
            out = np.concatenate([global_scaled, subject_scaled_part, untouched_part], axis=-1)
        else:
            out = np.concatenate([global_scaled, subject_transformed], axis=-1)
            
        return out.astype(np.float32)


class Log1pStandardScaler:
    """log1p, then global standardization, on the first ``scale_dim`` columns.

    Activity counts are zero-inflated and heavy-tailed; standardizing raw counts
    lets a few large bins dominate. log1p compresses the tail first. Columns past
    ``scale_dim`` (e.g. missing-data masks, time features) pass through unchanged.
    Statistics are global (fit on the training split), so no per-person level or
    spread is computed or encoded.
    """

    def __init__(self, scale_dim: Optional[int] = None, eps: float = 1e-8):
        self.scale_dim = scale_dim
        self.eps = eps
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def _cols(self, X: np.ndarray) -> int:
        return X.shape[1] if self.scale_dim is None else self.scale_dim

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Log1pStandardScaler":
        d = self._cols(X)
        Z = np.log1p(np.clip(X[:, :d].astype(np.float64), 0.0, None))
        self.mean_ = Z.mean(axis=0)
        self.scale_ = np.maximum(Z.std(axis=0), self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        d = self._cols(X)
        out = X.astype(np.float64, copy=True)
        out[:, :d] = (np.log1p(np.clip(out[:, :d], 0.0, None)) - self.mean_) / self.scale_
        return out.astype(np.float32)
