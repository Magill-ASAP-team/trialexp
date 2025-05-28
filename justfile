# Run the full pipeline
run-pipeline: copy-data
    # Run the Python file
    snakemake --snakefile workflow/pycontrol.smk -k -c20 --rerun-triggers mtime

copy-data:
    python workflow/scripts/00_create_session_folders.py

# search for the pattern in the data folders
find-session pattern:
  fd --type d {{pattern}} $SESSION_ROOT_DIR

find-data pattern:
  fd -e ppd -e tsv {{pattern}} $SESSION_ROOT_DIR 

# Search for and execute workflow in a session folder
make-session session_id:
  #!/usr/bin/bash 
  #shebang is necessary otherwise it will be executed line by line indepedently
  target=$(fd --type d {{session_id}} $SESSION_ROOT_DIR|head -n1)
  echo $target
  read -p "Are you sure you want to proceed? [y/N] " ans; \
  if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then \
      echo "Aborted."; exit 1; \
  fi
  snakemake --snakefile workflow/pycontrol.smk -c20 -F $target/processed/pycontrol_workflow.done

sort session_id:
    #!/usr/bin/bash 
    #shebang is necessary otherwise it will be executed line by line indepedently
    target=$(fd --type d {{session_id}} $SESSION_ROOT_DIR|head -n1)
    echo $target
    read -p "Are you sure you want to proceed? [y/N] " ans; \
    if [ "$ans" != "y" ] && [ "$ans" != "Y" ]; then \
        echo "Aborted."; exit 1; \
    fi
    snakemake --snakefile workflow/spikesort.smk -c20 -F $target/processed/spike_workflow.done

    