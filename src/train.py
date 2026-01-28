import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

class ModelTrainer:
    def __init__(self, model, output_dir='outputs'):
        self.model = model
        self.output_dir = output_dir
        self.history = None
        
    def setup_callbacks(self):
        os.makedirs(os.path.join(self.output_dir, 'models'), exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            os.path.join(self.output_dir, 'models', 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        )
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        return [checkpoint, early_stop, reduce_lr]
    
    def train(self, train_gen, val_gen, epochs=50, class_weights=None):
        callbacks = self.setup_callbacks()
        
        self.history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
