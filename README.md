Automated 2D Drafting System
<img width="840" alt="image" src="https://github.com/user-attachments/assets/d6777a7a-f7cb-4011-84b6-8e9a78311e82" />

1. Problem Statement & Goal


Problem Statement
➢ Dimensioning rules are company-specific and are developed through trial & error and F-costs, which reflect critical manufacturing know-how.
➢ Skilled engineers develop this expertise over time and use it to define key dimensions appropriate for the manufacturing environment, directly affecting F-costs in
new production development.
➢ When skilled engineers leave, companies suffer significant losses, and training replacements requires considerable time and cost.
➢ 3D CAD software does not include Auto dimensioning function based on geometry and only provides basic projection function.
➢ Only a function that allows dimensions entered in the 3D environment to be automatically added when drafting.

<img width="888" alt="image" src="https://github.com/user-attachments/assets/90999d6f-86a3-47ae-b6a2-c27fc8277029" />



▪ Goal: Dimensioned 2D drawings derived from 3D assembly models with no dimension data.
▪ General 2D generation method
(1) Projection (The process of converting 3D into 2D), (2) Dimensioning (The process of adding dimensions to 2D) (3) e.g. ( section, annotation )



2. Learning Data & Model ( Projection )

▪ Generate Dataset & Training
➢ Input Format : ① 3D CAD files: .stp, .step, ② 2D created by Engineer
➢ 2D Projection Generation : Automatically generate six orthogonal views: ①Front, ②Back, ③Top, ④Bottom, ⑤Left, ⑥Right from a 3D CAD model to use as
learning data for machine learning.
➢ Data : Compare 2D projections with 2D drawings created by engineers (Ground truth)
➢ By learning the projection views with high matching accuracy, the model is trained to apply similar projection views when encountering similar 3D shapes.


<img width="908" alt="image" src="https://github.com/user-attachments/assets/7df57c83-376b-4870-9469-2f8aad839db2" />




3. Learning Model

Learning Model for the 2D–3D Mapping [ GNN (Graph Neural Network) ]
➢ Convert the 3D STEP file into a graph representation to prepare for GNN-based learning.
➢ A GNN regression model is trained to learn from 3D graph inputs and view-matching scores derived in Step 1.
➢ Once trained, the model can predict matching scores for new 3D shapes, i.e., infer which 2D views are likely to appear in the drawing.
➢ Based on the predicted matching scores, the system can suggest optimal 2D drafting strategies for unseen 3D models.
<img width="1384" alt="image" src="https://github.com/user-attachments/assets/56cbd095-1c4b-4ff1-90cc-9428873d01ca" />



4. Results and Feature work ( Projection )

 ▪ Result
• Projection : The model successfully predicted the optimal view with average accuracy of 85%,
this suggests the system is capable of mimicking engineering judgment in view selection.
<img width="508" alt="image" src="https://github.com/user-attachments/assets/57b9f6a7-e642-48a6-945c-852526344bfc" />
<img width="508" alt="image" src="https://github.com/user-attachments/assets/b5f5bc48-2aac-46a2-be62-2b6e1b14a5e9" />

▪ Future Work
1. Problem: When the shape looks the same from left-right, top-bottom, or front-back, the model learns the same shape more than once with a high score. This
causes too many similar views and lowers the accuracy.
2. Improvement Idea: For X, Y, and Z direction views, we will check how similar they are. If two views are very similar, we will ignore one of them during training to
avoid duplicates.



2. Learning Data ( Dimensioning )

▪ Step 1 – Generate Dataset & Training
➢ DXF Feature & Relation Extraction : Geometric features such as lines, arcs, and circles and Ground truth dimension information is extracted from the drawing.
➢ The system analyzes the spatial relationships between entities and constructs integrated vectors that represent.
(a) individual feature characteristics, (b) relational and positional context, and (c) the corresponding true dimension values.
➢ These vectors are used to train a machine learning model, which is saved for future prediction tasks.

<img width="1339" alt="image" src="https://github.com/user-attachments/assets/bc5b0f91-d02c-4b6b-8e8c-3ec1e6747cce" />



3. Results and Feature work ( Dimensioning )
▪ Result
1. Dimensioning: The system achieved an accuracy of 60% in identifying and placing core dimensions.


<img width="1389" alt="image" src="https://github.com/user-attachments/assets/d2a49cd6-2749-4102-addd-7fb369873df8" />
2. To improve this relatively low accuracy, we replaced the initial encoder-decoder model with
a residual network model that includes positional information. However, the added positional
features did not contribute meaningfully to accuracy improvement.

<img width="1230" alt="image" src="https://github.com/user-attachments/assets/97822ca9-08f3-41e1-967e-0cd4595288c4" />

▪ Future Work
1. By using the positional data learned from the 3D model, we adopt a 3D-based learning approach instead of training on individual projection views.
To improve accuracy and avoid repeated dimensions that appear across multiple views, we aim to develop an integrated and consistent 2D drafting
system.


<img width="1230" alt="image" src="https://github.com/user-attachments/assets/102469bf-d2a7-4986-8851-9014839785a6" />






