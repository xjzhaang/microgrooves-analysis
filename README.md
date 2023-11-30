# Microgroove Movie Analysis

This repository contains code and resources for the analysis of microgroove movies, organized into four main folders: `preprocessing`, `segmentation`, `classification`, and `tracking`.

The workflow segments microgrooves and nuclei, then performs classification to distinguish between caged and uncaged nuclei in every frame of a movie. 
Finally, it opens a windowless Fiji and performs tracking on the nuclei through a Jython TrackMate script using Kalman filters. 

## Getting started


If you're new to the project, consider following these steps to get started:

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/xjzhaang/microgrooves-movies.git
   ```
   

2. **Create a Python 3.11 environment and install the dependencies:**

   Using conda (recommended):
   ```bash
   conda create --name micrgrooves-analysis python=3.11
   conda activate micrgrooves-analysis
   conda install --file requirements.txt
   ```
   Using pip:
   ```bash
   # Using venv (built-in module in Python 3.3 and newer)
   python3.11 -m venv micrgrooves-analysis
   microgroove-movies\Scripts\activate   #Windows OS
   source micrgrooves-analysis/bin/activate   #Linux/MAC OS
   pip install -r requirements.txt
   ```
   **If your device has a CUDA enabled GPU**: it is recommended to install Pytorch and CUDA from https://pytorch.org/get-started/locally/

   **You should also install Fiji** from https://imagej.net/software/fiji/downloads 

## Usage

### Finding your directory
All image outputs are saved in a folder named `output` within the repository. 
When using the scripts, you only need to specify `directory_path`, 
which can be the relative path to your images with respect to this repository.

For example in the following structure, your directory paths would be <em>../data/first_experiment</em> 
and <em>../data/second_experiment</em>:
```
data/
├── first_experiment/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── second_experiment/
│   ├── image1.tif
│   ├── image2.tif
│   └── ...
├── ....
│
microgrooves-analysis/
├── main.py
├── README.md
├── ....

```

### Running the code
Below are some commands you can use to perform the different steps. 

1. **Preprocessing**

   ```bash
   cd micrgrooves-analysis
   python main.py --preprocess -d [directory_path]
   ```

2. **Segmentation with Cellpose**

   ```bash
   python main.py --segment -d [directory_path]
   ```
   
3. **Classification**

   ```bash
   python main.py --classify -d [directory_path]
   ```
4. **Tracking with TrackMate**
   
   As we use PyImageJ to initialize a gateway interface to your locally installed Fiji/ImageJ2, you need 
to specify the absolute installation path of Fiji on your machine. Normally for linux, it is <em>/opt/fiji/Fiji.app</em>.
   ```bash
   python main.py --track -d [directory_path] -fiji [installation_path]
   ```