# Install poetry if it's not already installed
pip install poetry

# Install the dependencies and create a virtual environment
poetry install

# Activate the virtual environment created by poetry
source $(poetry env info --path)/bin/activate  # On Windows use `$(poetry env info --path)\Scripts\activate`

# Run  script
python src/lstm_auto_complete.py
