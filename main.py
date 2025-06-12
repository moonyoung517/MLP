import os
import glob

from model.train_test import train_model, test_model
from model.dataset import DXFGraphDataset, build_combined_data
from model.model import EnsembleMLP
from config import TRAIN_DXF_DIR, TEST_DXF_DIR, MODEL_SAVE_PATH, EPOCHS, RETRAIN

from dxf_parser.id_generator import IDGenerator


 
def main():

    print("Loading training data...")
    train_files = glob.glob(os.path.join(TRAIN_DXF_DIR, '*.dxf'))
    if not train_files:
        print("No training files found. Skipping training.")
        train_dataset = None
    else:

        if RETRAIN:
            train_dataset = DXFGraphDataset(train_files)

            if train_dataset:
                print("Training model...")
                train_model(train_dataset, MODEL_SAVE_PATH, epochs=EPOCHS)
            else:
                print("Cannot retrain: training dataset is empty.")
                
        elif not os.path.exists(MODEL_SAVE_PATH):
            if train_dataset:
                
                print(f"Model file not found. Training new model at {MODEL_SAVE_PATH}...")
                train_model(train_dataset, MODEL_SAVE_PATH, epochs=EPOCHS)
            else:
                print("No model found and no training data available to train a model.")
                return
        else:
            print(f"Using existing trained model: {MODEL_SAVE_PATH}")

    print("Loading test data...")
    test_files = glob.glob(os.path.join(TEST_DXF_DIR, '*.dxf'))
    if not test_files:
        print("No test files found. Aborting.")
        return

    print("Evaluating result for first test file...")    
    test_dataset = DXFGraphDataset(test_files)
    test_model(test_dataset, MODEL_SAVE_PATH)


if __name__ == "__main__":
    main()
