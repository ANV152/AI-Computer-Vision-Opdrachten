import numpy as np
from collections import Counter
train_data = np.genfromtxt('dataset1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
dates = np.genfromtxt( 'dataset1.csv', delimiter=';', usecols=[0])
def get_season_label(date):
    """Assigns a season label based on YYYYMMDD."""
    if date < 20000301:
        return 'winter'
    elif date < 20000601:
        return 'lente'
    elif date < 20000901:
        return 'zomer'
    elif date < 20001201:
        return 'herfst'
    else:
        return 'winter'
def get_season_label_v(date):
    """Assigns a season label based on YYYYMMDD."""
    if date < 20010301:
        return 'winter'
    elif date < 20010601:
        return 'lente'
    elif date < 20010901:
        return 'zomer'
    elif date < 20011201:
        return 'herfst'
    else:
        return 'winter'
# Create labels for all rows
y_train = np.array([get_season_label(date) for date in dates])


#x_vector = feature vector of example i
#


def get_min_max_dataset(dataset):
    """
    Berekent de minimale en maximale waarde voor elke kolom (attribuut) in een dataset.

    Args:
        dataset (numpy.ndarray): Een 2D NumPy array met numerieke waarden.

    Returns:
        tuple: Een tuple van twee lijsten. De eerste lijst bevat de minimale waarden
               per kolom, de tweede lijst de maximale waarden.
    """
    max_atr = []
    min_atr = []
    for i in range(0, len(dataset[0])):
        max_atr.append(max(dataset[:, i]))
        min_atr.append(min(dataset[:, i]))
    return min_atr, max_atr

def normalize_datasets(data_1, data_2):
    """
    Normaliseert 2 datasets door gebruik te maken van de min-max normalisatie

    Args:
        data_1 (numpy.ndarray): Eerste dataset.
        data_2 (numpy.ndarray): Tweede dataset.

    Returns:
        tuple: A tuple of the normalized datasets.
    """
    # min en max per kolom
    min_max_data1_ = get_min_max_dataset(data_1)
    min_max_data2_ = get_min_max_dataset(data_2)
    
    # globale min en max van iedere feature wordt in een np array opgeslagen
    min_ = np.minimum(min_max_data1_[0], min_max_data2_[0])
    max_ = np.maximum(min_max_data1_[1], min_max_data2_[1])
    # Hieronder nemen we voordeel van de volgende numpy array eigenschap
    range_ = max_ - min_
    
    # dit stukje code is delen door nul te voorkomen
    range_[range_ == 0] = 1
    
    # Normalize datasets
    data_1_normalized = (data_1 - min_) / range_
    data_2_normalized = (data_2 - min_) / range_

    return data_1_normalized, data_2_normalized
            
            

def k_NN(X_train, y_train, x_test, k=3):
    """
     Voer k-Nearest Neighbors uit en verminder k bij ties zonder recursie.

    Args:
        X_train(np.array): dataset zonder labels
        y_train(np.array): labels van X_train
        x_test(np.array): Test vector

    Returns:
        tuple: A tuple of the normalized datasets.
    """
    if k < 1:
        raise ValueError("k moet minimaal 1 zijn.")

    # Bereken de afstanden tussen de testvector en elke trainingsvector
    distances = [(np.linalg.norm(x_test - x_train), label) 
                 for x_train, label in zip(X_train, y_train)]
    
    # Sorteer de afstanden
    sorted_distances = sorted(distances, key=lambda x: x[0])
    
    """
    We hoeven niet opnieuw naar de vectoren in X_train te kijken, 
    omdat de afstand van een vector slechts wordt gebruikt om de dichtstbijzijnde buren te sorteren.
    Zodra   we de dichtstbijzijnde k-buren hebben, zijn alleen hun labels van belang 
    voor de classificatie, ongeacht hoe dicht ze bij het testpunt liggen.
    """
    # Itereer en probeer k-1 als er een tie is
    while k >= 1:
        k_closest_labels = [label for _, label in sorted_distances[:k]]
        label_counts = Counter(k_closest_labels).most_common()
        
        if len(label_counts) > 1 and label_counts[0][1] == label_counts[1][1]:
            k -= 1  #verminder k en probeer opnieuw
        else:
            #geen tie -> retourneer het label met de hoogste frequentie
            return label_counts[0][0]
    
    # Als k is verminderd tot 1, neem het label van de dichtstbijzijnde buur
    k_closest_labels = [label for _, label in sorted_distances[:k]]
    label_counts = Counter(k_closest_labels).most_common()
    return label_counts[0][0]

X_validation =np.genfromtxt('validation1.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})

dates_validation = np.genfromtxt('validation1.csv', delimiter=';', usecols=0, dtype=int)

# labels voor validation data
y_validation = np.array([get_season_label_v(date) for date in dates_validation])

#Normalisatie van de waardes in iedere dataset
train_data_normalize, X_validation_normalized = normalize_datasets(train_data, X_validation)
predictions_validation = [k_NN(train_data_normalize, y_train, x_val, k=2) for x_val in X_validation_normalized]

# Compute validation accuracy
accuracy_validation = np.mean([pred == true_label for pred, true_label in zip(predictions_validation, y_validation)])

print(f"Validation Accuracy: {(accuracy_validation * 100):.2f} %")




  


