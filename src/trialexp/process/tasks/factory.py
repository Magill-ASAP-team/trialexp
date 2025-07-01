from trialexp.process.tasks.base import TaskAnalysis
from trialexp.process.tasks.reaching_go_spout import ReachingGoSpoutAnalysis
from trialexp.process.tasks.default import DefaultTask
from pathlib import Path

def get_task_analysis(task_name: str, session_path: Path) -> TaskAnalysis:
    if task_name == 'reaching_go_spout_incr_break2_April24':
        return ReachingGoSpoutAnalysis(task_name, session_path)
    elif task_name == 'reaching_go_spout_incr_break2_nov22':
        return ReachingGoSpoutAnalysis(task_name, session_path)
    else:
        return DefaultTask(task_name, session_path)
