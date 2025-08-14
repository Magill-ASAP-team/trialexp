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
- `python -m pytest tests/` - Run test suite
- Individual test files: `python -m pytest tests/test_trialexp.py`

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
2. **PyControl Processing**: Multiple scripts (01-07) for behavioral analysis
3. **Spike Sorting**: Scripts in `workflow/scripts/spike_sorting/`
4. **DeepLabCut Integration**: Video analysis workflows
5. **Data Export**: Parquet format for visualization with PlotJuggler

### Key Components
- **Trial Dataset Classes**: Handle time-locked trial data with flexible filtering
- **Session Analysis**: Compute trial outcomes and success metrics
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
1. `params/tasks_params.csv` - Add task parameters
2. `src/trialexp/process/pycontrol/session_analysis.py` - Update `compute_success()` and `compute_trial_outcome()` functions
3. `workflow/scripts/07_time_warping.py` - Add time warping specifications

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