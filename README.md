# DBLP Dataset Final Project

This project aims to analyze the DBLP dataset to identify and visualize clusters
of research topics and examine how they change over time. Such an analysis could
reveal which fields have grown in popularity, which have declined, and how
different areas are interrelated.

## Project Structure

```bash
.
├── README.md                # This file
├── project.ipynb            # Main Jupyter notebook containing all code
├── pyproject.toml           # Project dependencies and configuration
├── datasets/                # Directory for dataset
├── figures/                 # Directory for generated figures
├── Report/                  # Directory Holding Report and Typst Code
└── table_results/           # Directory for generated tables
```

### Dependencies

- Python 3.8+
- uv

### Installation & Setup

#### Step 1: Install uv

If you dont have `uv` installed, go to their website to see the preferred
installation method based on your OS.

```link
https://docs.astral.sh/uv/getting-started/installation/
```

#### Step 2: Install dependencies

```bash
cd [project-directory]

uv venv .venv # Create virtual environment

source .venv/bin/activate # Activate virtual environment

uv sync # Install dependencies from pyproject.toml

uv pip show notebook # Verify notebook is installed
```

#### Step 3: Launch Jupyter Notebook

```bash
jupyter notebook # launch Jupyter server
```

#### Step 4: Open the Notebook

After running the above command, a localhost token should display in the
terminal. In this format:

```link
http://localhost:8888/?token=...
```

Copy and paste it into your web browser to access the Jupyter Notebook then open
`project.ipynb`.
