# Computer Vision System for Calculating the Elastic Modulus in Concrete

## Project Description
This project aims to develop a system based on computer vision and deep learning techniques to calculate the elastic modulus of concrete cylinders subjected to controlled loading conditions. The system automates the analysis of compression tests, reducing the reliance on expensive specialized equipment and minimizing errors associated with manual calculations.

## Context
The elastic modulus is a key parameter in civil engineering that describes a material's stiffness and its ability to deform under load. Traditionally, its measurement requires specialized equipment like compressometers-extensometers, which may not be accessible in resource-constrained laboratories.

This system proposes a more accessible and cost-effective approach, leveraging modern computer vision and deep learning techniques to obtain precise data efficiently.

## Key Features
- **Data Collection and Preprocessing**: Management of images from compression tests on concrete cylinders.
- **Deep Learning Model**: Implementation of a model to detect concrete cylinders and measure their deformation.
- **Automated Calculations**: Accurate computation of the elastic modulus based on the ASTM C469 standard.
- **Performance Evaluation**: Comparison of the system with traditional methods in terms of accuracy and efficiency.

## Requirements
- Python 3.8 or higher
- Required libraries:
  - OpenCV
  - PyTorch
  - NumPy
  - Matplotlib
  - Scikit-learn
  - pandas
- Additional software:
  - [LabelImg](https://github.com/heartexlabs/labelImg) for data labeling

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/user/elastic-modulus-project.git
   ```

2. Navigate to the project directory:
   ```bash
  cd DATASET/EntornoEntrenamiento/TESIS/yolov5
   ```

3. Install the dependencies:
   ```bash
  pip install -r requirements.txt
   ```

## Usage
1. Navigate to the elastic modulus script directory:
   ```bash
  ModuloDeElasticidad/
   ```

2. **Prepare the Input Data**: 
	- Add a folder containing the test videos you want to process into the project directory. 
	- Include the paths to these videos in the `main` function of the `elastic_modulus.py` file. 
	
3. **Configure the Output Directory**: 
	- Create an output folder where the system will save the results. Ensure this folder path is correctly set in the script. 

4. Run the main.py program:
   ```bash
 ModuloDeElasticidad/main.py 
   ```

5. **Analyze the Results**: 
	- After execution, the results (processed videos, calculations, and related outputs) will be saved in the specified output folder. You can analyze these files for further insights.
   ```bash
 "route parameter"/resultados/
   ```

## Additional Information
All details regarding the research, structure, and functionality of the program are documented in the accompanying thesis, titled **"Sistema de Visión Computacional para el Cálculo del Módulo de Elasticidad en Ensayos de Concreto (Versión 3)"**. The thesis, provided in PDF format, contains an in-depth explanation of the system's design, methodology, and implementation, offering a comprehensive resource for understanding the project.

