import numpy as np
import matplotlib.pyplot as plt
from Op_5_AAI_k_nearest_neighbour import *
def calculate_predictions_and_save(train_data_normalize, y_train, X_validation_normalized, y_validation, output_file):
    """
    Calculate predictions for various k values, save results to a file, and plot the prediction percentage.

    Args:
        train_data_normalize (numpy.ndarray): Normalized training data.
        y_train (numpy.ndarray): Labels for the training data.
        X_validation_normalized (numpy.ndarray): Normalized validation data.
        y_validation (numpy.ndarray): Labels for the validation data.
        output_file (str): File to save the results.
    """
    results = []

    # Iterate over odd k values from 1 to 65
    for k in range(1, 66):
        # Predict for all validation samples
        predictions_validation = [
            k_NN(train_data_normalize, y_train, x_val, k=k) for x_val in X_validation_normalized
        ]
        # Calculate prediction accuracy
        correct_predictions = sum(
            1 for pred, true in zip(predictions_validation, y_validation) if pred == true
        )
        prediction_percentage = (correct_predictions / len(y_validation)) * 100
        results.append((k, prediction_percentage))

    # Save results to a file
    with open(output_file, 'w') as f:
        f.write("k,Prediction_Percentage\n")
        for k, percentage in results:
            f.write(f"{k},{percentage:.2f}\n")

    # Extract data for plotting
    k_values = [k for k, _ in results]
    percentages = [percentage for _, percentage in results]

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, percentages, marker='o', linestyle='-', color='b')
    plt.title("Prediction Percentage vs. k Value")
    plt.xlabel("k Value")
    plt.ylabel("Prediction Percentage (%)")
    plt.grid(True)
    plt.show()

# Example usage (replace with actual data and function definitions):
calculate_predictions_and_save(train_data_normalize, y_train, X_validation_normalized, y_validation, "results.csv")
