
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
from datetime import datetime

class TaskAnalysis(ABC):
    """
    Abstract base class for task-specific analysis.
    Each task should have its own subclass that implements the abstract methods.
    """

    def __init__(self,  task_name: str, session_path: Path):
        self.session_path = session_path
        self.task_name = task_name

    @abstractmethod
    def process_pycontrol(self, df_pycontrol: pd.DataFrame, session_time: datetime, subjectID: str) -> pd.DataFrame:
        """
        Process the pycontrol data for the specific task.
        """
        pass

    @abstractmethod
    def run_behavioral_analysis(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame):
        """
        Run the behavioral analysis for the specific task.
        """
        pass

    @abstractmethod
    def plot_results(self, df_events: pd.DataFrame, df_conditions: pd.DataFrame):
        """
        Plot the results for the specific task.
        """
        pass
