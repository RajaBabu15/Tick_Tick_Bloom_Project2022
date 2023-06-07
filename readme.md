# Algal Bloom Detection using Satellite Imagery

This project aims to detect and classify cyanobacteria blooms in small, inland water bodies using satellite imagery. The objective is to leverage machine learning techniques to assist water quality managers in making informed decisions and allocating resources effectively.

## Contest Information

This project is based on the ML contest conducted by NASA. The contest focuses on utilizing satellite imagery data to detect and classify cyanobacteria blooms. By accurately identifying and classifying these blooms, the project aims to contribute to water quality management and environmental monitoring efforts.

### Contest Details

- Contest Name: Tick Tick Bloom: Harmful Algal Bloom Detection Challenge
- Hosted By: NASA

## Data Sources

The project utilizes the following data sources:

### Satellite Imagery

- Sentinel-2: Provides satellite imagery with a resolution of 10m. The Level-1C (L1C) data is obtained via Google Earth Engine or Microsoft Planetary Computer. Level-2A (L2A) data can also be obtained using these platforms.
- Landsat: Includes Landsat 8 and Landsat 9 data with a resolution of 30m. The data from March 2013 onwards is used, and Landsat 7 data from January and February 2013 is also considered.

### Climate Data

- High-Resolution Rapid Refresh (HRRR): Offers climate data with a resolution of 3km. This data is used for additional analysis and modeling.

### Elevation Data (not used)

- Copernicus DEM: Provides elevation data with a resolution of 30m. Although not utilized in this project, it may be relevant for future analysis.

## Project Structure

The project repository includes the following files and folders:

- `main.ipynb`: Jupyter Notebook containing the main code for data preprocessing, modeling, and generating predictions.
- `environment.yml`: Conda environment configuration file for reproducing the project environment.
- `data/`: Folder containing the necessary data files for the project.
- `models/`: Folder to store trained models or model checkpoints.
- `results/`: Folder to store the output predictions or evaluation results.

## Setup Instructions

To set up the project environment, follow these steps:

1. Install Anaconda from [https://www.anaconda.com](https://www.anaconda.com).

2. Clone this repository:

git clone <https://github.com/RajaBabu15/Tick_Tick_Bloom_Project2022>
cd Tick_Tick_Bloom_Project2022

3. Create a new conda environment:

conda env create -n ticktickbloom --file environment.yml

4. Activate the environment:

conda activate ticktickbloom


## Running the Project

1. Make sure the conda environment is activated (`conda activate ticktickbloom`).

2. Open the `main.ipynb` Jupyter Notebook.

3. Follow the instructions in the notebook to preprocess the data, train models, and generate predictions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

