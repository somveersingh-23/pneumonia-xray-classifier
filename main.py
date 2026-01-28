import os
import sys
# ...existing code...
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
# ...existing code...
from src.data_loader import ChestXrayDataLoader
from src.model import PneumoniaClassifier
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator

def main():
    DATA_DIR = 'data'
    OUTPUT_DIR = 'outputs'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    
    print("="*70)
    print("PNEUMONIA DETECTION FROM CHEST X-RAY IMAGES")
    print("="*70)
    
    print("\n[1/5] Loading and preprocessing data...")
    data_loader = ChestXrayDataLoader(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    train_gen, val_gen, test_gen = data_loader.create_generators()
    
    print(f"\nTraining samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Test samples: {test_gen.samples}")
    print(f"Class indices: {train_gen.class_indices}")
    
    class_weights = data_loader.compute_class_weights(train_gen)
    print(f"\nClass weights (to handle imbalance): {class_weights}")
    
    print("\n[2/5] Building ResNet50 transfer learning model...")
    classifier = PneumoniaClassifier(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        learning_rate=LEARNING_RATE
    )
    model = classifier.build_transfer_learning_model()
    print("\nModel architecture created successfully")
    
    print("\n[3/5] Training model...")
    trainer = ModelTrainer(model, OUTPUT_DIR)
    history = trainer.train(train_gen, val_gen, EPOCHS, class_weights)
    
    print("\n[4/5] Evaluating model...")
    evaluator = ModelEvaluator(model, OUTPUT_DIR)
    evaluator.plot_training_history(history)
    report, cm = evaluator.evaluate_model(test_gen)
    
    print("\n[5/5] Saving final results...")
    model.save(os.path.join(OUTPUT_DIR, 'models', 'final_model.h5'))
    print(f"\nAll outputs saved to '{OUTPUT_DIR}' directory")
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()
