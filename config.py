# [Final Modules: config.py, main.py with model reuse and visualization]

# File: config.py
TRAIN_DXF_DIR = "./dxfs"
TEST_DXF_DIR = "./test"
MODEL_SAVE_PATH = "importance_model.pt"
EPOCHS = 50
RETRAIN = False  # True: always retrain, False: reuse trained model if exists