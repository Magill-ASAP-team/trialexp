# Run the full pipeline
run-pipeline: copy-data
    # Run the Python file
    snakemake --snakefile workflow/pycontrol.smk -k -c20 --rerun-triggers mtime

copy-data:
    python workflow/scripts/00_create_session_folders.py
