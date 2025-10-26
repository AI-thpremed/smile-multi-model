# Preoperative Assessment and Prediction for SMILE Surgery

This project aims to provide a preoperative assessment and prediction tool for SMILE (Small Incision Lenticule Extraction) surgery. By integrating preoperative corneal curvature maps and multiple patient data from tables, we have achieved enhanced prediction performance.

## Project Overview

SMILE surgery is a popular refractive surgery for correcting myopia. However, there is often an unpredictable discrepancy between the programmed optical zone (POZ) and the final postoperative effective optical zone (EOZ), which can affect visual quality. To address this challenge, we developed a machine learning model that combines preoperative clinical parameters and corneal curvature maps to predict whether the postoperative EOZ diameter will be less than 5.5 mm.

## Methodology

- **Data Integration**: We collected preoperative parameters (such as age, gender, refractive error, corneal thickness, etc.) and anterior corneal curvature maps from patients who underwent SMILE surgery.
- **Model Design**: A multimodal machine learning model was developed. It integrates the preoperative parameters and corneal curvature maps through a conservative embedding and fusion approach. The goal is to keep the model simple and effective, considering the limited data volume.
- **Prediction Performance**: The model demonstrated superior predictive performance compared to models based on single-data modalities. It achieved balanced performance across different datasets, with high accuracy and stability in predicting the postoperative EOZ size.

## Key Features

- **Enhanced Prediction**: The integration of corneal curvature maps and preoperative parameters significantly improves the prediction accuracy of postoperative EOZ size.
- **Simplicity and Efficiency**: The conservative embedding and fusion scheme ensures the model remains simple and easy to implement, without requiring complex structures.
- **Clinical Relevance**: The model provides valuable insights for personalized surgical planning, helping surgeons make more informed decisions and potentially improving visual outcomes.

## Usage

To use this project, you can clone the repository. Please refer to the `main-embedding.py` script and the table examples in the `Label` folder for detailed information on data processing and model training.

## Contributions

Contributions to this project are welcome! If you have any suggestions or improvements, please feel free to submit a pull request or open an issue. We appreciate your support in enhancing the prediction tool for SMILE surgery.

## Acknowledgements

We thank all the patients and surgeons involved in the study for their support. Special thanks to the research team for their dedication and hard work.
