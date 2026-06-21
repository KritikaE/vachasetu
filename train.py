import os
import torch
from src.data_pipeline import generate_synthetic_dataset
from src.model import VachaSetuNet, train_model

def main():
    print("=== Vacha-Setu Model Training ===")
    
    # 1. Generate synthetic dataset
    print("Generating synthetic lip landmarks dataset...")
    X, y = generate_synthetic_dataset(num_samples_per_class=150, num_frames=30)
    print(f"Dataset generated. Shape of X: {X.shape}, Shape of y: {y.shape}")
    
    # 2. Initialize the model
    model = VachaSetuNet(
        input_dim=120,      # 40 landmarks * 3 coordinates
        latent_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_classes=5
    )
    
    # 3. Train the model
    print("Starting training...")
    history = train_model(
        model=model,
        X=X,
        y=y,
        epochs=25,
        batch_size=16,
        lr=0.001,
        val_split=0.2
    )
    
    # 4. Save the trained weights
    model_path = "vachasetu_model.pth"
    print(f"Saving model weights to {model_path}...")
    torch.save(model.state_dict(), model_path)
    print("Training complete and model saved successfully!")

if __name__ == "__main__":
    main()
