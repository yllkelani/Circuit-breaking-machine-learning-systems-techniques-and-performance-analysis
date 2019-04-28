from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from sklearn.neural_network import MLPClassifier

class cb:
    def __init__(self):
        self.alert = 0

    # kNN
    def tune_knn(self, X, k = 3, noise_prop = 0.05):
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        distances, indices = nbrs.kneighbors(X)
        avg_distances = np.mean(distances, axis=1)
        sorted_avg = sorted(enumerate(avg_distances), key=lambda x:x[1])
        percentile = round(len(X) * (1 - noise_prop))
        self.knn_thres = sorted_avg[percentile][1]
        norm_indices = [x[0] for x in sorted_avg[:percentile]]
        abnorm_indices = [x[0] for x in sorted_avg[percentile:]]
        return norm_indices, abnorm_indices

    def train_knn(self, X, k = 3, tune=False):
        self.nbrs = NearestNeighbors(n_neighbors=k).fit(X)

    def test_knn(self, x):
        distances, indices = self.nbrs.kneighbors(x.values.reshape([1, -1]))
        avg_dis = np.mean(distances)

        if avg_dis > self.knn_thres:
            self.alert = 1

        return avg_dis
        
    # DBSCAN
    def tune_dbscan(self, X, noise_prop = 0.05):
        check_max = np.max((X.max()).values)
        check_min = np.min((X.min()).values)
        step = np.mean(X.std().values)
        while(True):
            for s in np.arange(check_min if check_min > 0 else step, check_max, step):
                dbs = DBSCAN(s)
                dbs_labels = dbs.fit_predict(X)
                label_noise = (len([label for label in dbs_labels if label == -1]))
                ratio = (label_noise / len(dbs_labels))
                if ratio <= noise_prop and ratio > (noise_prop/5 * 4):
                    self.dbscan_s = s
                    norm_indices = [x[0] for x in enumerate(dbs_labels) if x[1] != -1]
                    abnorm_indices = [x[0] for x in enumerate(dbs_labels) if x[1] == -1]
                    return norm_indices, abnorm_indices

            # If range doesn't find a suitable parameter
            if ratio > 0:
                check_min = check_max
                check_max *= 2
            else:
                check_min = step
                step /= 2

    def train_dbscan(self, X):
        self.dbscan = DBSCAN(self.dbscan_s)
        self.dbscan.fit(X)

    def test_dbscan(self, X, x):
        dbs_labels = self.dbscan.fit_predict(X.append(x))
        if dbs_labels[-1] == -1:
            self.alert = 1

        return dbs_labels[-1]

    # Neural Network
    def tune_nn(self, X, y):
        self.mlp = MLPClassifier(random_state=42).fit(X, y) # 1 hidden layer, 100 neurons
        return self.mlp.predict(X)

    def train_nn(self, X, y):
        self.mlp = MLPClassifier(random_state=42).fit(X, y)

    def test_nn(self, X):
        if self.mlp.predict(X) == 1:
            self.alert = 1
    def reset(self):
        self.alert = 0
