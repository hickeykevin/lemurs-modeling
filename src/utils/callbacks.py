from lightning import Callback
import pandas as pd

class LabelHistoryCallback(Callback):
    """Callback to print the chronological label history for a specific user.
    
    This helps verify that the persistence baseline is correctly identifying 
    the sequence of survey responses.
    
    Args:
        target_user_id (int): The ID of the participant to track.
    """
    
    def __init__(self, target_user_id: int = 27):
        super().__init__()
        self.target_user_id = target_user_id
        
    def on_train_start(self, trainer, pl_module):
        dm = trainer.datamodule
        
        # Ensure datamodule has loaded the data
        if not hasattr(dm, 'master_df') or dm.master_df is None:
            dm.setup()
            
        df = dm.master_df
        
        available_users = sorted(df['app_user_id'].unique())
        
        if self.target_user_id not in available_users:
            print(f"\n[LabelHistoryCallback] User ID {self.target_user_id} not found.")
            print(f"[LabelHistoryCallback] Available User IDs (first 20): {available_users[:20]}")
            return
            
        # Filter and sort by timestamp
        user_data = df[df['app_user_id'] == self.target_user_id].sort_values('record_timestamp')
        
        print(f"\n[LabelHistoryCallback] Full Label History for User {self.target_user_id}:")
        print("-" * 50)
        # We display the timestamp and the target answer
        print(user_data[['record_timestamp', 'answer']].to_string(index=False))
        print("-" * 50)
        print(f"Total entries for user: {len(user_data)}\n")
