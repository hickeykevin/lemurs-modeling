from typing import Any, Dict, List, Optional
import numpy as np
from copy import deepcopy

class SubjectScaler:
    """Scaler wrapper that applies standardizing/scaling on a per-subject level.
    
    This class wraps an arbitrary scikit-learn scaler and maintains a separate 
    fitted instance for each unique subject/user ID. If a user is not seen during 
    training, a scaler copy is dynamically fit on their data on the fly.
    """
    def __init__(self, base_scaler: Any):
        self.base_scaler = base_scaler
        self.scalers: Dict[Any, Any] = {}
        self.global_scaler: Optional[Any] = None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "SubjectScaler":
        """Standard fit method for compatibility, fitting a single global scaler."""
        self.global_scaler = deepcopy(self.base_scaler)
        self.global_scaler.fit(X, y)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standard transform method for compatibility, applying the global scaler."""
        if self.global_scaler is not None:
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
        self.global_scaler.fit(stacked_all)
        
        # Fit per-subject scalers
        unique_users = np.unique(user_ids)
        for uid in unique_users:
            user_seqs = [sequences[i] for i, u in enumerate(user_ids) if u == uid]
            user_stacked = np.concatenate(user_seqs, axis=0)  # [N_u * T, F]
            
            scaler = deepcopy(self.base_scaler)
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
        out = np.zeros_like(seqs_np)
        
        unique_users = np.unique(user_ids)
        for uid in unique_users:
            user_mask = (user_ids == uid)
            
            # If user has not been seen before, fit standardizer dynamically on the fly
            if uid not in self.scalers:
                user_seqs = seqs_np[user_mask]  # [N_u, T, F]
                user_stacked = user_seqs.reshape(-1, f)
                
                scaler = deepcopy(self.base_scaler)
                scaler.fit(user_stacked)
                self.scalers[uid] = scaler
                
            # Retrieve the subject-specific scaler and transform
            scaler = self.scalers[uid]
            user_seqs = seqs_np[user_mask]  # [N_u, T, F]
            u_n, u_t, u_f = user_seqs.shape
            
            transformed = scaler.transform(user_seqs.reshape(-1, f)).reshape(u_n, u_t, u_f)
            out[user_mask] = transformed
            
        return out.astype(np.float32)
