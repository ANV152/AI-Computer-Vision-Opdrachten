import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

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

# Normaliseer de data
def normalize_data(data):
    """Normaliseer de data in de database op basis van de berekende max en min waardes

    Args:
        data (np.array  ): Dataset 

    Returns:
        np.array: genormaliseerde dataset
    """
    min_values = data.min(axis=0)
    max_values = data.max(axis=0)
    return (data - min_values) / (max_values - min_values)
            
def k_means_clustering(X, k, max_iters=100, tol=1e-4):
    """
    Voer k-Means clustering uit.

    Args:
        X (numpy.ndarray): De dataset, een 2D-array met vorm (n_samples, n_features).
        k (int): Aantal clusters.
        max_iters (int): Maximum aantal iteraties.
        tol (float): Tolerantie voor veranderingen in centroiden.

    Returns:
        tuple: (clusters, centroids)
            clusters (list): Lijst van clusterindices voor elke datapunten.
            centroids (numpy.ndarray): De k-centrumwaarden.
    """
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]

    for iteration in range(max_iters):
        #toewijzing van punten aan dichtstbijzijnde centroid
        clusters = []
        for point in X:
            distances = np.linalg.norm(point - centroids, axis=1)
            cluster_index = np.argmin(distances)
            clusters.append(cluster_index)
        
        clusters = np.array(clusters)
        
        # update centroiden als het gemiddelde van de punten in elke cluster
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(k)])

        
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break  # stop als de verandering in centroiden onder de tolerantie is
        centroids = new_centroids

    return clusters, centroids

def determine_optimal_k(X):
    """
    Bepaal het optimale aantal clusters (k) met een scree plot.

    Args:
        X (numpy.ndarray): De dataset, een 2D-array met vorm (n_samples, n_features).

    Returns:
        None (maakt een scree-plot).
    """
    distortions = []
    K = range(1, 11)  # Test k van 1 tot 10
    for k in K:
        _, centroids = k_means_clustering(X, k)
        distortions.append(np.sum(np.min(np.linalg.norm(X[:, None] - centroids, axis=2), axis=1)))

    plt.figure(figsize=(8, 5))
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('Aantal clusters k')
    plt.ylabel('Distortion (inertia)')
    plt.title('Scree-plot om optimale k te bepalen')
    plt.show()

#maximum vote -->  clusters
def assign_season_to_clusters(clusters, labels):
    """Deze functie bepaalt welk label het beste de betekenis van elk cluster.

    Args:
        clusters (np.numpy): Array die clusterindeling voor elk datapunt bevat
        labels (list): seizoenlabels in dit geval

    Returns:
        Dictionary: Dictionary waarbij de sleutel het cluster is en de waarde het meest voorkomende label
    """
    cluster_labels = {}
    for cluster in range(k):
        cluster_points = labels[clusters == cluster]
        most_common = Counter(cluster_points).most_common(1)[0][0]
        cluster_labels[cluster] = most_common
    return cluster_labels

train_data = np.genfromtxt('dataset1.csv', delimiter=';', 
                           usecols=[1,2,3,4,5,6,7], 
                           converters={5: lambda s: 0 if s == b"-1" else float(s), 
                                       7: lambda s: 0 if s == b"-1" else float(s)})
dates = np.genfromtxt( 'dataset1.csv', delimiter=';', usecols=[0])
# Create labels for all rows
y_train = np.array([get_season_label(date) for date in dates])


k = 4

train_data_normalized = normalize_data(train_data)

clusters, centroids = k_means_clustering(train_data_normalized, k)

cluster_labels = assign_season_to_clusters(np.array(clusters), y_train)

# Resultaten
for cluster, season in cluster_labels.items():
    print(f"Cluster {cluster} is toegewezen aan seizoen: {season}")

# Optimal k bepalen met scree plot
determine_optimal_k(train_data_normalized)

  


