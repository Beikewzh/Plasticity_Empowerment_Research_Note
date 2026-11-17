#!/bin/bash
# Run the exact checks of the note on a Mila CPU node.
cd "$SCRATCH/note_checks_v2" || exit 1
python3 --version
python3 verify_note.py
