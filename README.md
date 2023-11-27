markdown

# Microgroove Movie Analysis

This repository contains code and resources for the analysis of microgroove movies, organized into three main folders: `preprocessing`, `segmentation`, and `classification`.

The workflow segments microgrooves and nuclei, then performs classification to distinguish between caged and uncaged nuclei in every frame of a movie.

## Getting started


If you're new to the project, consider following these steps to get started:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/xjzhaang/microgrooves-movies.git
   ```
   

2. **Create a Python 3.11 environment and install the dependencies:**

   Using conda (recommended):
   ```bash
   conda create --name microgroove-movies python=3.11
   conda activate microgroove-movies
   conda install --file requirements.txt
   ```
   Or if you want to go the slow way:
   ```bash
   # Using venv (built-in module in Python 3.3 and newer)
   python3.11 -m venv microgroove-movies 
   microgroove-movies\Scripts\activate   #Windows OS
   source microgroove-movies/bin/activate   #Linux/MAC OS
   pip install -r requirements.txt
   ```
   **If your device has a CUDA enabled GPU**: it is recommended to install Pytorch and CUDA from https://pytorch.org/get-started/locally/
