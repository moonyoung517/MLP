# Automated 2D Drafting System 

## 1. Problem Statement
### Creating 2d drawings from 3d data is a time-consuming task

➢ Dimensioning rules are company-specific and are developed through trial & error and F-costs, which reflect critical manufacturing know-how.

➢ Skilled engineers develop this expertise over time and use it to define key dimensions appropriate for the manufacturing environment, directly affecting F-costs in new production development.

➢ When skilled engineers leave, companies suffer significant losses, and training replacements requires considerable time and cost.




## 2. Project Goal
### From 3D assembly models without dimension data, Create 2d drawing that automatically generates dimensions

➢ It is divided into two steps.

(1) Projection (The process of converting 3D into 2D that best represents 3D)

(2) Dimensioning (The process of predicting dimensions to 2D)

![image](https://github.com/user-attachments/assets/9e30e317-f449-4e44-b0e3-61541542d043)


## 2 - 1. Learning Data & Method (Projection Step)

➢ Input Format : 3D CAD files: .stp, .step

➢ Output Format : 2D DXF files with separete view

➢ Method 
1) Automatically generate six orthogonal views from a 3D CAD model by CAD APIs(①Front, ②Back, ③Top, ④Bottom, ⑤Left, ⑥Right)
2) By learning the projection views with high matching accuracy, the model is trained to apply similar projection views when encountering similar 3D shapes.
3) learning data for machine learning.

The process of converting 3D into 2D that best represents 3D

➢ Learning Model 

Learning Model for the 2D–3D Mapping [ GNN (Graph Neural Network) ]
➢ Convert the 3D STEP file into a graph representation to prepare for GNN-based learning.
➢ A GNN regression model is trained to learn from 3D graph inputs and view-matching scores derived in Step 1.
➢ Once trained, the model can predict matching scores for new 3D shapes, i.e., infer which 2D views are likely to appear in the drawing.
➢ Based on the predicted matching scores, the system can suggest optimal 2D drafting strategies for unseen 3D models.

![image](https://github.com/user-attachments/assets/6748ed03-928e-47f0-90aa-ed6d5c94b660)


➢ Results and Feature work ( Projection )

 ▪ Result
• Projection : The model successfully predicted the optimal view with average accuracy of 85%,
this suggests the system is capable of mimicking engineering judgment in view selection.

<img width="508" alt="image" src="https://github.com/user-attachments/assets/57b9f6a7-e642-48a6-945c-852526344bfc" />

▪ Future Work
1. Problem: When the shape looks the same from left-right, top-bottom, or front-back, the model learns the same shape more than once with a high score. This
causes too many similar views and lowers the accuracy.
2. Improvement Idea: For X, Y, and Z direction views, we will check how similar they are. If two views are very similar, we will ignore one of them during training to
avoid duplicates.


## 2 - 2. Learning Data & Method (Dimensioning)

➢ Input Format : 2D AutoCAD File: .dxf(result from projection step)

➢ Output Format : 2D Dimension Prediction results

➢ Method 

1) DXF Feature & Dimension Extraction : Geometric features such as lines, arcs, and circles and Ground truth dimension(Y) information is extracted from the drawing.
   
2) Relation Definition : Using the feature data(start, end point, nominal value, type, etc..)
   
3) Define Integrated vector : Adding feature and relation data(we use integrated vector as dataset X)

4) Train model using integrated vector and ground truth
  
5) Save model for future prediction tasks.

<img width="1339" alt="image" src="https://github.com/user-attachments/assets/bc5b0f91-d02c-4b6b-8e8c-3ec1e6747cce" />

➢ Learning Model : ResidualMLP + ResidualAttentionMLP

➢ Result
1. Dimensioning: The system achieved an accuracy of 60% in identifying and placing core dimensions.

<img width="1389" alt="image" src="https://github.com/user-attachments/assets/d2a49cd6-2749-4102-addd-7fb369873df8" />

2. To improve this relatively low accuracy, we replaced the initial encoder-decoder model witha residual network model that includes positional information. However, the added positional
features did not contribute meaningfully to accuracy improvement.


➢ Future Work
1. By using the positional data learned from the 3D model, we adopt a 3D-based learning approach instead of training on individual projection views.
To improve accuracy and avoid repeated dimensions that appear across multiple views, we aim to develop an integrated and consistent 2D drafting
system.


<img width="1230" alt="image" src="https://github.com/user-attachments/assets/102469bf-d2a7-4986-8851-9014839785a6" />


## Quick Start
The full pipeline consists of two stages:
(1) 6-view projection learning & inference → (2) 2D dimension importance prediction

### Create Conda Environment

```bash
# Using conda
conda env create -f environment.yaml
conda activate mlp

or

# (Optional) Using mamba for faster installation
conda install -n base -c conda-forge mamba
mamba env create -f environment.yaml
mamba activate mlp
```


### Run Full Pipeline (Projection + Dimension Prediction)

```
# Execute both stages sequentially
python -m main.py
```

### 6-View Matching Score Prediction

Predict view-to-view matching scores based on STEP and DXF files using a GNN model trained on 6 orthographic projections.

**Main Components:**

- `dataset.py`, `model.py`: PyTorch Geometric-based training pipeline
- `extract_dxf_edges.py`: DXF edge visualization and feature extraction
- `train.py`, `eval.py`: GNN model training and evaluation

**Step-by-step Execution:**

```bash
cd model3d_6_view_pred

# 1. Extract 6-view projections from STEP
python -m 1_extract_dxf.py

# 2. Compute matching scores between views
python -m 2_matching_dxf.py

# 3. Train GNN model on matching scores
python -m 3_learning.py

# 4. Run inference using trained GNN model
python -m 4_main.py
```

---

### 2D Dimension Importance Prediction

Predict the importance of dimension features in DXF drawings using a pre-trained classification model (`importance_model.pt`).

**Main Components:**

- `dxf_parser/`: DXF parser and feature extractor
- `model/`: PyTorch model for dimension classification
- `main.py`: Prediction entry point for a given DXF file

**Usage:**

```bash
cd cad2d_dimension_pred

python -m main.py
```

### Folder Structure
```
Project Root

├── environment.yml                  # Conda environment configuration file
├── main.py                          # Entry point to run the full pipeline
├── config.py                        # Global configuration
├── README.md                        # Project overview and documentation

├── cad2d_dimension_pred/            # Module for predicting dimension importance from 2D DXF drawings
│   ├── main.py                      # Main script for 2D inference pipeline
│   ├── config.py                    # Module-specific configuration
│   ├── requirements.txt             # Dependency list
│   ├── importance_model.pt          # Trained MLP model for importance prediction
│   ├── example.dxf                  # Sample drawing file
│   ├── test_code.py                 # Unit test script
│   ├── tempCodeRunnerFile.py        # Temporary file (can be removed)
│   ├── dxfs2/                       # DXF drawing dataset (multi-view)
│   │   ├── *.dxf
│   │   └── ...                      # Various DXF views (Front, Top, etc.)
│   ├── test/                        # Test files for evaluation
│   │   ├── *.dxf
│   │   └── ...
│   ├── dxf_parser/                  # DXF parsing and dimension extraction utilities
│   │   ├── dimensions.py
│   │   ├── dimension_matcher.py
│   │   ├── feature_extractor.py
│   │   ├── id_generator.py
│   │   ├── relations.py
│   │   ├── units.py
│   │   ├── utils.py
│   │   ├── visualization.py
│   │   └── __init__.py
│   ├── model/                       # MLP model training and inference
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── train_test.py
│   │   └── __init__.py

├── model3d_6_view_pred/             # Module for 3D CAD model multi-view score prediction
│   ├── 1_extract_dxf.py             # Extract DXF features
│   ├── 2_matching_dxf.py            # Match STEP and DXF views
│   ├── 3_learning.py                # Train GNN model
│   ├── 4_main.py                    # (Possible duplicate or script stub)
│   ├── config.py
│   ├── test_predictions_new_preprocessing.csv
│   ├── data/                        # STEP and DXF datasets
│   │   ├── *.step, *.dxf
│   │   ├── step_to_dxf_matching_data_new_preprocessing.xlsx
│   │   └── test/                   # STEP test files for inference
│   │       ├── *.step
│   │       └── image/              # (empty or placeholder)
│   ├── dxf_parser/                  # Utilities for DXF extraction and matching
│   │   ├── extract_dxf_fn.py
│   │   ├── matching_dxf_fn.py
│   │   ├── visualization.py
│   │   └── __init__.py
│   ├── model/                       # GNN model and learning functions
│   │   ├── best_model_fold_preprocessing*.pth
│   │   ├── learning_fn.py
│   │   └── __init__.py

├── test/                            # Shared test files and evaluation results
│   ├── *.step, *.dxf                # Combined test shapes and drawings
│   ├── all_predictions.csv          # Aggregated prediction results
│   └── image/                       # PNG visualization for each view (front, top, side, etc.)

└── README.md                        # Project description and documentation

```
