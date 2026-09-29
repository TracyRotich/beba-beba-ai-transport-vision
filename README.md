# Beba Beba AI: Public Transport Computer Vision

A team capstone exploring image classification, object detection and a Streamlit prototype for Kenyan public-service vehicle monitoring.

## Problem

Manual identification and monitoring of public-service vehicles is difficult to scale. This project investigates how computer vision could support fleet operators, transport planners and enforcement teams by classifying vehicle types, detecting vehicles in street imagery and presenting results through an application prototype.

The work is a technical proof of concept. It does not determine legal compliance or roadworthiness without human review and additional validated data.

## System components

| Component | Purpose | Approach |
|---|---|---|
| Vehicle classification | Classify PSV images by type or fleet category | Custom CNNs, MobileNetV2 and EfficientNetV2 |
| Vehicle detection | Locate boda-bodas, buses, matatus and tuk-tuks in images | YOLOv8s and YOLOv8m |
| Application prototype | Accept an uploaded image and return model output | Streamlit and exported Keras models |

## Reported results

- The fleet-classification experiments report approximately **93% validation accuracy** for the selected EfficientNetV2 approach.
- The selected YOLOv8m detector reports **mAP@50 of 0.823**, **recall of 0.746**, and approximately **9.3 ms inference time** in the notebook environment.
- YOLOv8m improved recall over YOLOv8s, while the experiments also show weaker performance for some classes and at stricter localisation thresholds.

These figures come from the current development notebooks and small project datasets. They are not production benchmarks.

## Tracy Rotich's contribution

This was a group project. The Git history records Tracy's work on the Objective 3 fleet-classification notebook, deployment code, repository integration and documentation. Other team members contributed to the shared classification and detection objectives; their commits remain in the repository history.

## Repository guide

- [`Objective1_Capstone.ipynb`](Objective1_Capstone.ipynb): initial PSV image-classification workflow.
- [`Objective2_Capstone.ipynb`](Objective2_Capstone.ipynb): YOLOv8 vehicle detection and model comparison.
- [`Objective3_Capstone.ipynb`](Objective3_Capstone.ipynb): fleet classification using CNN and transfer-learning approaches.
- [`my-streamlit-app/app.py`](my-streamlit-app/app.py): image-upload application prototype.
- [`Capstone Final Presentation .pdf`](Capstone%20Final%20Presentation%20.pdf): stakeholder presentation.

## Technical workflow

1. Validate, rename and standardise image data.
2. Explore class balance, dimensions, aspect ratios and brightness.
3. Apply stratified train/validation/test splitting and augmentation.
4. Compare custom CNN and transfer-learning classifiers.
5. Train and evaluate YOLOv8 detectors across four vehicle classes.
6. Export selected models and connect them to a Streamlit interface.

## Tools

Python, TensorFlow, Keras, EfficientNetV2, MobileNetV2, Ultralytics YOLOv8, OpenCV, Albumentations, scikit-learn, Streamlit and Jupyter.

## Run the Streamlit prototype

```bash
cd my-streamlit-app
pip install -r requirements.txt
streamlit run app.py
```

The model files are stored in the application directory. Training notebooks were developed in Google Colab and may require path updates before rerunning locally.

## Limitations and responsible use

- The datasets are small and may not represent changes in geography, lighting, camera angle, vehicle design or fleet branding.
- Validation accuracy from a limited sample can overstate performance on new operating conditions.
- Class-level recall and localisation quality require more testing, especially for underrepresented vehicles.
- Predictions should not be used for enforcement, insurance or compliance decisions without independent validation, human review, privacy assessment and a documented appeals process.

## Next iteration

Add licensed data and annotation documentation, create a truly external test set, report per-class precision and recall with confidence intervals, measure end-to-end latency, document failure cases, add automated tests for preprocessing, and package inference behind a reproducible service interface.
