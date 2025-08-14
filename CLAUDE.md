# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Triexap is a neuroscience data analysis pipeline for processing pycontrol (behavioral), pyphotometry (fiber photometry), and Neuropixels (electrophysiology) data. The pipeline uses Snakemake workflows with environment-based configuration and supports multiple cohorts and experimental sessions.

## Development Commands

### Environment Setup
- `uv sync` - Install dependencies (use `uv sync --extra full` for GPU and Snakemake support)
- Environment variables must be configured in a `.env` file at project root

### Pipeline Execution (via just recipes)
- `just run-pipeline` - Run the complete analysis pipeline  
- `just copy-data` - Copy raw data into session folders
- `just make-session <search_term> [FLAGS]` - Process a specific session (e.g., `just make-session TT021-2025-06-04 -F`)
- `just sort <search_term> [FLAGS]` - Run spike sorting workflow for a session
- `just find-session <pattern>` - Search for session directories
- `just find-data <pattern>` - Search for data files (.ppd, .tsv)

### Testing
- `uv run python -m pytest tests/` - Run test suite
- Individual test files: `uv run python -m pytest tests/test_trialexp.py`

### Direct Snakemake Usage
- `uv run snakemake --snakefile workflow/pycontrol.smk -k -c20 --rerun-triggers mtime` - PyControl workflow
- `uv run snakemake --snakefile workflow/spikesort.smk -k -c20` - Spike sorting workflow

## Architecture

### Core Package Structure
- `src/trialexp/` - Main package containing all analysis modules
- `src/trialexp/dataset_classes/` - Data structures for trial-based analysis
  - `Trials_Dataset` class and subclasses for continuous/event data
  - Metadata handling with trial filtering and condition management
- `src/trialexp/process/` - Processing modules organized by data type:
  - `pycontrol/` - Behavioral data analysis
  - `pyphotometry/` - Fiber photometry processing  
  - `ephys/` - Electrophysiology data handling
  - `deeplabcut/` - Video analysis integration
  - `anatomy/` - Anatomical data processing

### Workflow System
- Snakemake-based pipeline with three main workflows:
  - `workflow/pycontrol.smk` - Behavioral data processing
  - `workflow/spikesort.smk` - Spike sorting pipeline  
  - `workflow/Snakefile` - Master workflow combining both
- Session-based processing with automatic target generation
- Cohort filtering: currently processes `2025_June_cohort` and `2025_July_cohort`

### Configuration Management
- Environment-based configuration via `.env` file
- `src/trialexp/config.py` - Central configuration loading
- Session directory structure: `{cohort}/by_sessions/{task}/{animal_session_id}/`
- Configuration parameters in `params/` directory:
  - `animal_info.csv` - Animal metadata
  - `tasks_params.csv` - Task-specific parameters
  - `data_format.csv` - Data format specifications

### Data Processing Flow
1. **Session Creation**: `workflow/scripts/00_create_session_folders.py`
2. **PyControl Processing**: Processor-based workflow scripts for behavioral analysis
   - `01_process_pycontrol.py` - Complete session processing using `BehavioralProcessor.process_full_session()`
   - `02_plot_pycontrol_data.py` - Session visualization using `BehavioralProcessor.generate_session_plots()`
   - `03_export_parquet.py` - Data export using `BehavioralProcessor.process_export_data()`
   - `04_import_pyphotometry.py` - Photometry processing using `BehavioralProcessor.process_photometry_session()`
   - `06_behavorial_analysis.py` - Behavioral metrics using `BehavioralProcessor.compute_behavioral_metrics()`
   - `07_time_warping.py` - Time warping analysis using `BehavioralProcessor.process_time_warping()`
3. **Spike Sorting**: Scripts in `workflow/scripts/spike_sorting/`
4. **DeepLabCut Integration**: Video analysis workflows
5. **Data Export**: Parquet format for visualization with PlotJuggler

### Processor Architecture (NEW)
The pipeline uses a modular processor-based architecture for task-specific analysis:

#### Processor Classes
- **PyControlProcessor** (`src/trialexp/process/pycontrol/processors/pycontrol_processor.py`)
  - Core behavioral data processing: session parsing, trial extraction, condition computation
  - Task-specific success criteria and outcome computation
  - Basic behavioral plotting and event distribution analysis

- **PhotometryProcessor** (`src/trialexp/process/pycontrol/processors/photometry_processor.py`)
  - Photometry data import, preprocessing, and alignment with PyControl
  - Event-locked signal extraction and time warping analysis
  - Photometry-specific visualization and validation

- **BehavioralProcessor** (`src/trialexp/process/pycontrol/processors/behavioral_processor.py`)
  - **DEFAULT PROCESSOR** - Combines PyControl and Photometry functionality via multiple inheritance
  - Complete workflow orchestration for end-to-end analysis
  - Session plotting, behavioral metrics, and data export

#### Processor Usage
```python
# Load processor (typically done automatically by workflow scripts)
from trialexp.process.pycontrol.processors import get_processor
processor = get_processor('BehavioralProcessor')  # Default for all tasks

# Available processor types
processors = list_processors()  # ['BaseTaskProcessor', 'PyControlProcessor', 'PhotometryProcessor', 'BehavioralProcessor']
```

#### Task Configuration
All tasks in `params/tasks_params.csv` are configured to use `BehavioralProcessor` by default. To use a different processor:
1. Modify the `processor_class` column in `params/tasks_params.csv`
2. Register custom processors using the factory system

#### Key Processing Methods
- `process_full_session()` - Complete PyControl session processing
- `process_photometry_session()` - Photometry import, alignment, and processing  
- `process_time_warping()` - Time warping analysis with task-specific configurations
- `generate_session_plots()` - Comprehensive behavioral visualization
- `compute_behavioral_metrics()` - Core behavioral analysis metrics

### Key Components
- **Trial Dataset Classes**: Handle time-locked trial data with flexible filtering
- **Processor Architecture**: Modular, inheritance-based system for task-specific analysis
- **Session Analysis**: Compute trial outcomes and success metrics via processors
- **Time Warping**: Synchronize data across different acquisition systems
- **Multi-modal Integration**: Combine behavioral, photometry, and ephys data

## Adding New Data

### New Cohort
Modify these files:
1. `params/animal_info.csv` - Add animal information
2. `workflow/scripts/00_create_session_folders.py` - Update session creation logic  
3. `workflow/pycontrol.smk` - Update cohort list in `session2analyze()`

### New Task
Modify these files:
1. `params/tasks_params.csv` - Add task parameters and specify `processor_class` (use `BehavioralProcessor` for standard tasks)
2. For custom task logic, create a new processor class inheriting from `BehavioralProcessor` and override specific methods:
   - `compute_trial_outcome()` - Custom outcome computation logic
   - `compute_success()` - Custom success criteria  
   - `get_timewarp_config()` - Custom time warping specifications
3. `params/timewarp_spec.json` - Add time warping specifications for new task types

## Development with Processor Architecture

### Creating Custom Processors
To create a task-specific processor:

```python
# Example: Custom processor for a new task type
from trialexp.process.pycontrol.processors import BehavioralProcessor, register_processor

class MyCustomProcessor(BehavioralProcessor):
    def compute_trial_outcome(self, df_events, df_conditions, task_config):
        # Custom outcome logic here
        df_conditions = super().compute_trial_outcome(df_events, df_conditions, task_config)
        # Add custom modifications
        return df_conditions

# Register the custom processor
register_processor("MyCustomProcessor", MyCustomProcessor)
```

### Testing Processors
```bash
# Test processor functionality
uv run python -c "
from trialexp.process.pycontrol.processors import get_processor, list_processors
print('Available processors:', list_processors())
processor = get_processor('BehavioralProcessor')
print('Processor loaded:', type(processor))
"

# Test specific processor methods
uv run python -c "
from trialexp.process.pycontrol.processors import get_processor
processor = get_processor('BehavioralProcessor')
print('Available methods:', [m for m in dir(processor) if not m.startswith('_')])
"
```

### Processor Development Guidelines
- Use `BehavioralProcessor` as base class for most custom processors
- Override specific methods rather than rewriting entire workflows
- Test processor registration and method inheritance
- Update `params/tasks_params.csv` to specify custom processor class
- Use `uv run python` for all testing commands

## Development Tools

### Web Applications
- `tools/figure_viewer/` - Streamlit app for figure visualization
- `tools/signal_viewer/` - Reflex-based signal dashboard
- Run with respective `app.py` files

### Specialized Scripts
- `tools/variable_extractor/` - Extract variables from data files
- `extern_scripts/` - External analysis scripts and templates

## Environment Variables Required

Key environment variables that must be set in `.env`:
- `SESSION_ROOT_DIR` - Root directory for session data
- `ETTIN_MOUNT_PATH` - Mount path for data storage
- `RAW_DATA_ROOT_DIR` - Raw data directory
- Additional paths for spike sorting, video analysis, and anatomy processing