import cclustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def doPCA(X, labels, n_dataset):
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    fig = px.scatter(components, x = 0, y = 1, title = 'Blobs', color=labels, labels={'0': 'PC 1', '1': 'PC 2'})
    fig.update_layout(
        width = 600,
        height = 600,
        title = 'Dataset nr {0} - samples = {1} - features = {2} - classes = {3}'.format(n_dataset, X.shape[0], X.shape[1], np.max(labels) + 1))
    fig.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1)
    fig.show()

def scaling(samples):
    scaler = StandardScaler()
    samples = scaler.fit_transform(samples)
    return samples

def create_dataset_base(samples, features, centers, standard_deviation_cluster = 1, standard = True, display = False, n_dataset = 0):
    # X = The generated samples
    # l = The integer labels for cluster membership of each sample
    X, l = make_blobs(n_samples = samples, n_features = features, centers = centers, cluster_std = standard_deviation_cluster, random_state = None)
    if standard:
        X = scaling(X)
    if display:
        doPCA(X, l, n_dataset)
    return X, l

sample0, l0 = create_dataset_base(samples = 3, features = 4, centers = 2, display = False, n_dataset = 0)
sample1, l1 = create_dataset_base(samples = 1000, features = 30, centers = 2, display = False, n_dataset = 1)
sample2, l2 = create_dataset_base(samples = 4000, features = 7, centers = 10, display = False, n_dataset = 2)
sample3, l3 = create_dataset_base(samples = 7000, features = 30, centers = 10, display = False, n_dataset = 3)
sample4, l4 = create_dataset_base(samples = 10000, features = 2048, centers = 5, display = False, n_dataset = 9)
sample5, l5 = create_dataset_base(samples = 15000, features = 2048, centers = 5, display = False, n_dataset = 10)

listOfDataset = []
listOfDataset.append((sample0, l0))
# listOfDataset.append((sample1, l1))
# listOfDataset.append((sample2, l2))
# listOfDataset.append((sample3, l3))
# listOfDataset.append((sample4, l4))
# listOfDataset.append((sample5, l5))

counter = 0
for eachTuple in listOfDataset:
    counter += 1
    
    print("////////////////DATABASE {0}////////////////".format(counter))
    thetacpu = cclustering.CircleClustering(dataset = eachTuple[0], target = eachTuple[1], precision = "medium", hardware = "cpu")

    print(thetacpu)