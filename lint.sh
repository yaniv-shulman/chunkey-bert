#!/bin/bash

LKB_REPO_DIR=$(git rev-parse --show-toplevel)

while getopts ":f" option; do
   case $option in
      f) # display Help
         FIX=1
         ;;
     \?) # Invalid option
         echo "Error: Invalid option, use -f to fix fixable reported problems"
         exit;;
   esac
done


if [ -z "$FIX" ]; then
    python -m black --check "$LKB_REPO_DIR"
    python -m ruff check "$LKB_REPO_DIR"
else
    python -m black "$LKB_REPO_DIR"
    python -m ruff check "$LKB_REPO_DIR" --fix
fi

python -m mypy "$LKB_REPO_DIR"
