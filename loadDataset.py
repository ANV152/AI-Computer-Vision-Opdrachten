import random
class_mapping = {
    'Iris-setosa': [1, 0, 0],
    'Iris-versicolor': [0, 1, 0],
    'Iris-virginica': [0, 0, 1]
}
def shuffle_and_split(dataset, train_ratio=0.8):
    """
    Shuffle the dataset and split it into training and test datasets.
    
    Args:
        dataset (list of lists): Each sublist represents a row with features and class label.
        train_ratio (float): Proportion of data to use for training (default is 0.8).
        
    Returns:
        tuple: A tuple containing the training set and test set.
    """
    random.seed(0)
    random.shuffle(dataset)
    
    # split index
    split_index = int(len(dataset) * train_ratio)
    
    # Split into training and test sets
    train_data = dataset[:split_index]
    test_data = dataset[split_index:]
    
    return train_data, test_data
def load_dataset(file_path):
    """
    Load the iris dataset from a file.
    
    Args:
        file_path (str): Path to the iris.data file.
    
    Returns:
        list: A list of tuples, where each tuple contains input features (list of floats) 
              and the one-hot encoded class label (list of ints).
    """
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Skip empty lines
                row = line.split(',')  # Split the line by commas
                # Convert numeric features to float
                features = [float(x) for x in row[:-1]]
                # Convert class label to one-hot vector
                label = class_mapping[row[-1]]
                dataset.append((features, label))
    return dataset
# Example Usage
file_path = "iris/iris.data"  
dataset = load_dataset(file_path)

# Shuffle and split the dataset
train_data, test_data = shuffle_and_split(dataset, train_ratio=0.8)