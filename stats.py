import json
import matplotlib.pyplot as plt

def plot_losses():
    try:
        with open("loss_history.json", "r") as f:
            loss_history = json.load(f)
    except FileNotFoundError:
        print("loss_history.json not found. Please train the model first.")
        return

    # Extract the data
    steps = [item[0] for item in loss_history]
    train_losses = [item[1] for item in loss_history]
    test_losses = [item[2] for item in loss_history]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, label="Train Loss", color="blue", linewidth=1.5)
    plt.plot(steps, test_losses, label="Test Loss", color="orange", linewidth=1.5)
    
    # Add labels and formatting
    plt.title("Transformer Loss over Training Steps")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save and display
    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=300)
    print("Plot saved successfully to loss_curve.png")
    
    # If you run this in an environment with a display, it will show the plot window
    plt.show()

if __name__ == "__main__":
    plot_losses()
