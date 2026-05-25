from Op_5_AAI_k_nearest_neighbour import *
test_dataset = np.genfromtxt('days.csv', delimiter=';', usecols=[1,2,3,4,5,6,7], converters={5: lambda s: 0 if s == b"-1" else float(s), 7: lambda s: 0 if s == b"-1" else float(s)})
dates = np.genfromtxt( 'dataset1.csv', delimiter=';', usecols=[0])

output_file = "knn_results.txt"

test_dataset_normalized, train_data_norm = normalize_datasets(test_dataset, train_data)

results = []
for x_test in test_dataset_normalized:
    prediction = k_NN(train_data_norm, y_train, x_test, k=61)
    results.append(prediction)

with open(output_file, "w") as file:
    for i, result in enumerate(results):
        file.write(f"Testvector {i+1}: Voorspeld label = {result}\n")

print(f"De resultaten zijn opgeslagen in {output_file}.")