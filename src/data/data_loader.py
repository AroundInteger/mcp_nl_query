"""
Data loader module for handling different sports data formats.
"""
import pandas as pd
from pathlib import Path
from typing import Union, Dict, List, Optional


class SportsDataLoader:
    """Class for loading and preprocessing sports data."""
    
    def __init__(self, data_dir: Union[str, Path]):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_match_data(self, 
                       sport: str, 
                       league: str, 
                       season: Optional[str] = None) -> pd.DataFrame:
        """
        Load match data for a specific sport and league.
        
        Args:
            sport: Sport name (e.g., 'RUGBY_UNION')
            league: League name (e.g., 'URC')
            season: Optional season identifier
            
        Returns:
            DataFrame containing match data
        """
        # Construct file path
        filename = f"{sport}_{league}"
        if season:
            filename += f"_{season}"
        filename += ".csv"
        
        file_path = self.raw_data_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"No data file found at {file_path}")
        
        # Load the data
        df = pd.read_csv(file_path)
        
        # Basic preprocessing
        df = self._preprocess_data(df)
        
        return df
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform basic preprocessing on the data.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        # Remove any duplicate rows
        df = df.drop_duplicates()
        
        # Convert date columns to datetime if they exist
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            df[col] = pd.to_datetime(df[col])
        
        # Handle missing values
        df = df.fillna(method='ffill')  # Forward fill for time series data
        
        return df
    
    def save_processed_data(self, 
                          df: pd.DataFrame, 
                          sport: str, 
                          league: str, 
                          season: Optional[str] = None) -> None:
        """
        Save processed data to the processed data directory.
        
        Args:
            df: DataFrame to save
            sport: Sport name
            league: League name
            season: Optional season identifier
        """
        filename = f"{sport}_{league}"
        if season:
            filename += f"_{season}"
        filename += "_processed.csv"
        
        file_path = self.processed_data_dir / filename
        df.to_csv(file_path, index=False) 