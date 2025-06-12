# Automated 2D Drafting System 
<img width="840" alt="image" src="https://github.com/user-attachments/assets/d6777a7a-f7cb-4011-84b6-8e9a78311e82" />

## 1. Problem Statement
### Creating 2d drawings from 3d data is a time-consuming task

➢ Dimensioning rules are company-specific and are developed through trial & error and F-costs, which reflect critical manufacturing know-how.

➢ Skilled engineers develop this expertise over time and use it to define key dimensions appropriate for the manufacturing environment, directly affecting F-costs in new production development.

➢ When skilled engineers leave, companies suffer significant losses, and training replacements requires considerable time and cost.

![image](https://github.com/user-attachments/assets/aea2bf06-5079-4be7-8ed3-3bd66572db66)

## 2. Project Goal
### From 3D assembly models without dimension data, Create 2d drawing that automatically generates dimensions

➢ It is divided into two steps.
(1) Projection (The process of converting 3D into 2D), (2) Dimensioning (The process of adding dimensions to 2D)



## 2 - 1. Learning Data & Method (Projection Step)

➢ Input Format : 3D CAD files: .stp, .step

➢ Method 
1) Automatically generate six orthogonal views from a 3D CAD model by CAD APIs(①Front, ②Back, ③Top, ④Bottom, ⑤Left, ⑥Right)
2) By learning the projection views with high matching accuracy, the model is trained to apply similar projection views when encountering similar 3D shapes.
3) learning data for machine learning.

<img width="908" alt="image" src="https://github.com/user-attachments/assets/7df57c83-376b-4870-9469-2f8aad839db2" />

➢ Learning Model 

Learning Model for the 2D–3D Mapping [ GNN (Graph Neural Network) ]
➢ Convert the 3D STEP file into a graph representation to prepare for GNN-based learning.
➢ A GNN regression model is trained to learn from 3D graph inputs and view-matching scores derived in Step 1.
➢ Once trained, the model can predict matching scores for new 3D shapes, i.e., infer which 2D views are likely to appear in the drawing.
➢ Based on the predicted matching scores, the system can suggest optimal 2D drafting strategies for unseen 3D models.

<img width="1384" alt="image" src="https://github.com/user-attachments/assets/56cbd095-1c4b-4ff1-90cc-9428873d01ca" />

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



# Installation


# 예시 코드
