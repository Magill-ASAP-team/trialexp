# Task Processing Refactor Plan

## Overview

This document outlines the plan to refactor the behavior task processing system in trialexp to provide more flexibility and maintainability. The current hardcoded approach in `session_analysis.py` will be replaced with a processor-based architecture.

## Current State

### Issues with Current Implementation
- **Hardcoded Logic**: Task-specific logic embedded in `compute_trial_outcome()` and `compute_success()` functions
- **Maintainability**: Adding new tasks requires modifying core functions with if/elif chains
- **Code Duplication**: Similar logic patterns repeated across ~15 task variants
- **Testing Difficulty**: Task-specific logic intermingled with general processing code
- **Scalability**: No clear extension mechanism for complex task variants

### Current Architecture
```
src/trialexp/process/pycontrol/
├── session_analysis.py           # Contains hardcoded task logic
├── common.py                     # General utilities
└── task_specific/               # Limited task-specific extensions
    └── *.py                     # Additional analysis scripts
```

## Target Architecture

### Design Principles
1. **Simple Configuration**: Tasks specify processor class in config file only
2. **Clean Inheritance**: Override only methods that need customization
3. **Backward Compatibility**: All existing tasks work with `BaseTaskProcessor`
4. **Extensibility**: Easy to add new processors for new task types

### New Architecture
```
src/trialexp/process/pycontrol/
├── session_analysis.py           # Updated to use processor factory
├── common.py                     # Unchanged
├── processors/
│   ├── __init__.py
│   ├── base.py                   # BaseTaskProcessor with current logic
│   ├── factory.py                # Processor registry and factory
│   ├── reaching.py               # Reaching task processors
│   ├── detection.py              # Detection task processors
│   └── incremental.py            # Incremental break processors
└── task_specific/               # Unchanged - additional analysis
    └── *.py
```

## Implementation Plan

### Primary Refactor Target: Workflow Scripts

**Main Focus**: Refactor `workflow/scripts/` files starting with `01_process_pycontrol.py` and working sequentially through each script file. Each script must be fully working before proceeding to the next.

**Key Principle**: Processors return processed data at file save breakpoints. Scripts handle all file I/O for Snakemake tracking. **No direct file saving in BaseTaskProcessor**.

### Phase 1: Foundation Setup
1. **Create processor directory structure**
   ```bash
   mkdir -p src/trialexp/process/pycontrol/processors
   touch src/trialexp/process/pycontrol/processors/__init__.py
   ```

2. **Create BaseTaskProcessor**
   - Move existing `compute_success()` logic from `session_analysis.py`
   - Move existing `compute_trial_outcome()` logic from `session_analysis.py`
   - Add standard processing pipeline method
   - **IMPORTANT**: Return processed data objects, do NOT save files directly

3. **Create processor factory**
   - Registry system for processor classes
   - Factory method to instantiate processors by name

4. **Update configuration**
   - Add `processor_class` column to `params/tasks_params.csv`
   - Set all existing tasks to use `BaseTaskProcessor`

### Phase 2: Script-by-Script Refactor
**Process each workflow script individually, ensuring it works before moving to next:**

1. **Start with `workflow/scripts/01_process_pycontrol.py`**
   - Identify file save points (`.to_parquet()`, `.to_csv()`, etc.)
   - Extract processing logic before save points into processor methods
   - Update script to call processor and handle returned data
   - Test script end-to-end with sample data
   - Verify all output files are created correctly

2. **Continue with subsequent scripts in order:**
   - `workflow/scripts/02_*.py`
   - `workflow/scripts/03_*.py`
   - etc.

3. **For each script:**
   - Map file save operations as processor method boundaries
   - Extract relevant processing logic to processor
   - Maintain script's file I/O responsibilities
   - Test thoroughly before proceeding

### Phase 3: Core Integration
1. **Update session_analysis.py**
   - Replace hardcoded functions with processor factory calls
   - Maintain existing function signatures for backward compatibility

2. **Test backward compatibility**
   - Ensure all existing tasks work with `BaseTaskProcessor`
   - Run existing test suite to verify no regressions

### Phase 4: Task-Specific Processors
1. **Identify task families** based on current hardcoded logic:
   - Reaching tasks (go_spout variants)
   - Detection tasks (go_nogo, discrimination)
   - Incremental break tasks
   - Navigation/maze tasks

2. **Create specialized processors**
   - Extract task-specific logic into processor subclasses
   - Override only necessary methods

3. **Update task configurations**
   - Assign appropriate processor classes to tasks
   - Test each processor with its target tasks

### Phase 5: Testing and Validation
1. **Create processor tests**
   - Unit tests for each processor class
   - Integration tests with real session data

2. **Performance validation**
   - Ensure no performance regression
   - Compare processing times before/after refactor

3. **Data validation**
   - Verify identical outputs for existing tasks
   - Compare processed data files before/after refactor

## File Changes

### New Files to Create
```
src/trialexp/process/pycontrol/processors/
├── __init__.py                   # Package initialization
├── base.py                       # BaseTaskProcessor class
├── factory.py                    # Processor registry and factory
├── reaching.py                   # Reaching task processors
├── detection.py                  # Detection task processors
└── incremental.py                # Incremental break processors
```

### Files to Modify
```
workflow/scripts/01_process_pycontrol.py            # PRIMARY TARGET - First script to refactor
workflow/scripts/02_*.py                            # Subsequent scripts in order
workflow/scripts/03_*.py                            # ...and so on
params/tasks_params.csv                             # Add processor_class column
src/trialexp/process/pycontrol/session_analysis.py  # Use processor factory
tests/                                              # Add processor tests
```

### Files to Preserve
```
src/trialexp/process/pycontrol/common.py            # Unchanged
src/trialexp/process/pycontrol/task_specific/       # Unchanged
```

### Script Refactor Strategy
Each workflow script will be modified to:
1. **Identify file save breakpoints** - Where `.to_parquet()`, `.to_csv()`, etc. occur
2. **Extract processing logic** - Move data processing before save points to processor methods
3. **Call processor methods** - Replace extracted logic with processor calls
4. **Handle returned data** - Scripts receive processed data and save to files
5. **Maintain Snakemake compatibility** - All file I/O remains in scripts for dependency tracking

## Configuration Changes

### Enhanced tasks_params.csv
Add new column `processor_class` with default value `BaseTaskProcessor`:

```csv
task,triggers,events,conditions,trial_window,extra_trigger_events,trial_parameters,processor_class
reaching_go_spout_bar_nov22,Go,,,[-2.0,6.0],,,BaseTaskProcessor
reaching_go_spout_bar_VR_April24,Go,,,[-2.0,6.0],,,ReachingVRProcessor
reaching_go_spout_incr_break2_nov22,Go,,,[-2.0,6.0],,,IncrementalBreakProcessor
detection_go_nogo_aug22,Go,,,[-1.0,4.0],,,DetectionProcessor
```

## Code Structure Examples

### BaseTaskProcessor
```python
class BaseTaskProcessor:
    def process_session(self, df_events, df_conditions, task_config):
        """Main processing pipeline - returns data, does NOT save files"""
        df_conditions = self.compute_conditions(df_events, df_conditions, task_config)
        df_conditions = self.compute_success(df_events, df_conditions, task_config)
        df_conditions = self.compute_trial_outcome(df_events, df_conditions, task_config)
        return df_conditions  # Return for script to save
    
    def compute_success(self, df_events, df_conditions, task_config):
        """Default success logic (moved from session_analysis.py)"""
        # Current hardcoded logic goes here
        return df_conditions  # Always return processed data
    
    def compute_trial_outcome(self, df_events, df_conditions, task_config):
        """Default outcome logic (moved from session_analysis.py)"""
        # Current hardcoded logic goes here
        return df_conditions  # Always return processed data
```

### Specialized Processor Example
```python
class ReachingVRProcessor(BaseTaskProcessor):
    def compute_trial_outcome(self, df_events, df_conditions, task_config):
        """VR-specific outcome logic"""
        # Override only this method for VR-specific behavior
        pass
```

### Factory Usage in Scripts
```python
# In workflow/scripts/01_process_pycontrol.py (example)
def main():
    # Load data
    df_events = load_events(input_file)
    df_conditions = load_conditions()
    task_config = get_task_config(task_name)
    
    # Process with appropriate processor
    processor_class_name = task_config.get('processor_class', 'BaseTaskProcessor')
    processor = get_processor(processor_class_name)
    
    # Get processed data (processor does NOT save files)
    processed_conditions = processor.process_session(df_events, df_conditions, task_config)
    
    # Script handles file saving for Snakemake tracking
    processed_conditions.to_parquet(output_file)
```

## Migration Strategy

### Incremental Approach
1. **Start with BaseTaskProcessor** - Move all existing logic, ensure no regressions
2. **Add factory integration** - Update session_analysis.py to use factory
3. **Create specialized processors one by one** - Extract task-specific logic gradually
4. **Update configurations** - Assign processor classes to tasks
5. **Validate each step** - Test after each major change

### Rollback Plan
- Keep original `session_analysis.py` functions as backup
- Use feature flags to switch between old and new systems during transition
- Maintain comprehensive test coverage throughout migration

## Testing Strategy

### Unit Tests
- Test each processor class in isolation
- Mock data inputs and verify outputs
- Test inheritance behavior

### Integration Tests
- Process real session data with each processor
- Compare outputs with existing system
- Performance benchmarks

### Regression Tests
- Ensure all existing tasks continue to work
- Verify identical outputs for BaseTaskProcessor tasks
- Test edge cases and error conditions

## Success Criteria

### Functional Requirements
- [ ] All existing tasks process successfully with BaseTaskProcessor
- [ ] Task-specific processors produce expected outputs
- [ ] No performance degradation
- [ ] Configuration changes work correctly

### Non-Functional Requirements
- [ ] Code is more maintainable and extensible
- [ ] New tasks can be added without modifying core functions
- [ ] Clear separation of concerns
- [ ] Comprehensive test coverage

## Future Enhancements

### Possible Extensions
1. **Pipeline Steps**: Break processors into composable pipeline steps
2. **Plugin System**: Dynamic loading of external processor classes
3. **Configuration Validation**: Schema validation for processor configurations
4. **Performance Optimization**: Caching and parallel processing

### New Task Support
With this architecture, adding new tasks becomes:
1. Create new processor class (if needed)
2. Register in factory
3. Add to tasks_params.csv
4. Test and deploy

## Timeline Estimate

- **Phase 1 (Foundation)**: 1-2 days
- **Phase 2 (Integration)**: 1 day
- **Phase 3 (Specialization)**: 2-3 days
- **Phase 4 (Testing)**: 1-2 days

**Total**: 5-8 days for complete refactor

## Notes

- This refactor maintains full backward compatibility
- All existing workflows and scripts continue to work unchanged
- The change is internal to the processing system only
- Configuration files are enhanced but not breaking changes