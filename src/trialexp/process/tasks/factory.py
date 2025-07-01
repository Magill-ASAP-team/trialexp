from trialexp.process.tasks.base import TaskAnalysis
from trialexp.process.tasks.reaching_go_spout import ReachingGoSpoutAnalysis


def get_task_analysis(task_name: str, **kwargs) -> TaskAnalysis:
    if task_name == 'reaching_go_spout_incr_break2_April24':
        return ReachingGoSpoutAnalysis(**kwargs)
    elif task_name == 'reaching_go_spout_incr_break2_nov22':
        return ReachingGoSpoutAnalysis(**kwargs)
    else:
        raise ValueError(f'Task {task_name} not supported')
