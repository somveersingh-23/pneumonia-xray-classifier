from src.data_loader import ChestXrayDataLoader

DATA_DIR = 'data'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Testing data loading...")
data_loader = ChestXrayDataLoader(DATA_DIR, IMG_SIZE, BATCH_SIZE)
train_gen, val_gen, test_gen = data_loader.create_generators()

print(f"\n✅ Training samples: {train_gen.samples}")
print(f"✅ Validation samples: {val_gen.samples}")
print(f"✅ Test samples: {test_gen.samples}")
print(f"✅ Class indices: {train_gen.class_indices}")

class_weights = data_loader.compute_class_weights(train_gen)
print(f"✅ Class weights: {class_weights}")

print("\n✅ Data loading successful! Ready for training.")
