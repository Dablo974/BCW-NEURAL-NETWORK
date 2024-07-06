import numpy as np

def kmeanspp_init(data, k):
    centroids = np.empty((k, data.shape[1]))
    centroids[0] = data[np.random.choice(data.shape[0])]

    for i in range(1, k):
        distances = np.min(np.linalg.norm(data - centroids[:i][:, np.newaxis], axis=2), axis=0)
        probabilities = distances / np.sum(distances)
        centroids[i] = data[np.random.choice(data.shape[0], p=probabilities)]

    return centroids

class ClusteringModel:
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.labels = None
        self.centroids = None
        self.data = None
        self.max_iter = 1000
        self.fuzzycoef = 2
        self.epsilon = 1e-6

    def fit(self, data):
        self.data = data
        U = np.random.rand(data.shape[0], self.num_clusters)
        U /= U.sum(axis=1, keepdims=True)
        V = None

        def calculate_prototype(U, X):
            m = self.fuzzycoef
            V = np.dot((U ** m).T, X) / (np.sum(U ** m, axis=0, keepdims=True).T + self.epsilon)
            return V

        def calculate_membership(U, D):
            power = 2 / (self.fuzzycoef - 1)
            D_t = D.T
            D_t[np.all(V == 0, axis=1)] = np.inf
            D = D_t.T
            D_sum = (D ** power).sum(axis=1)[:, np.newaxis]
            tmp = (D ** power) / (D_sum + self.epsilon)
            return tmp

        def calculate_dist(X, V):
            D = np.linalg.norm(X[:, np.newaxis, :] - V[np.newaxis, :, :], axis=2) ** 2
            return D
        iterator = 0
        for iteration in range(self.max_iter):
            iterator += 1
            V = calculate_prototype(U, data)
            D = calculate_dist(data, V)
            new_U = calculate_membership(U, D)
            if np.linalg.norm(new_U - U) < self.epsilon:
                break
            U = new_U

        self.centroids = V
        self.labels = np.argmax(U, axis=1)


    def predict(self, data):
        distances = np.linalg.norm(data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :], axis=2)
        return np.argmin(distances, axis=1)

    def silhouette_score(self, data):
        def dist(a, b):
            return np.linalg.norm(a - b)

        def a(i):
            cluster_points = data[self.labels == self.labels[i]]
            if len(cluster_points) == 1:
                return 0
            else:
                bool_idx = np.logical_and(self.labels == self.labels[i], np.arange(data.shape[0]) != i)
                if np.sum(bool_idx) == 0:
                    return 0
                else:
                    dists = np.linalg.norm(data[i] - data[bool_idx], axis=1)
                    return np.mean(dists)

        def b(i):
            min_dist = float('inf')
            for j in range(self.num_clusters):
                if j != self.labels[i]:
                    cluster_points = data[self.labels == j]
                    min_dist = min(min_dist, np.mean([dist(data[i], point) for point in cluster_points]))
            return min_dist

        silhouette_coefficients = [(b(i) - a(i)) / max(a(i), b(i)) for i in range(data.shape[0])]
        print("notation: ", np.mean(silhouette_coefficients))
        return np.mean(silhouette_coefficients)

    def compute_representation(self, X):
        """
        Compute the new representation of the data where each data point is
        represented by its distances to the centroids of the clusters.
        """
        # Compute the distances between each data point and each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids[np.newaxis, :], axis=2)

        # Use the distances as the new representation of the data
        return distances

# class ClassificationModel:

#     def __init__(self, input_dim, output_dim):
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.epochs = 100
#         self.learning_rate = 0.1
#         self.depth = 3

#         self.hidden_layers_z = np.zeros((self.depth, input_dim))
#         self.hidden_layers = np.zeros((self.depth, input_dim))
#         self.last_layer_z = np.zeros((1, output_dim))
#         self.last_layer = np.zeros((1, output_dim))

#         self.hidden_b = np.zeros((self.depth, input_dim))
#         self.last_b = np.zeros((1, output_dim))

#         self.hidden_w = np.zeros((self.depth, input_dim, input_dim))
#         self.last_w = np.zeros((output_dim, input_dim))

#     # init weights
#     def init_xavier(self):
#         x = np.sqrt(6 / (self.input_dim + self.input_dim))
#         self.hidden_w[0] = np.random.uniform(-x, x, size = (self.input_dim, self.input_dim))
#         for i in range(1, self.depth):
#             x = np.sqrt(6 / (2 * self.input_dim))
#             self.hidden_w[i] = np.random.uniform(-x, x, size = (self.input_dim, self.input_dim))
#         x = np.sqrt(6 / (self.input_dim + self.output_dim))
#         self.last_w = np.random.uniform(-x, x, size = (self.input_dim, self.output_dim))
#         print("INIT", self.hidden_w)

#     # x is an attribute here
#     def sigmoid(self, x):
#         return 1 / (1 + np.exp(-x))

#     def sigmoid_prime(self, x):
#         sig = self.sigmoid(x)
#         return sig * (1 - sig)

#     def relu(self, x):
#         return np.max(0, x)

#     def loss(self, predicted, expected):
#         return np.square(predicted - expected)

#     def loss_prime(self, predicted, expected):
#         return 2 * (predicted - expected)

#     def normalise(self, X):
#         return (X - X.mean()) / X.std()

#     # x represents an example here
#     def propagate(self, x):
#         self.hidden_layers_z[0] = x @ self.hidden_w[0] + self.hidden_b[0]
#         self.hidden_layers[0] = self.sigmoid(self.hidden_layers_z[0])
#         for i in range(1, self.depth):
#             self.hidden_layers_z[i] = self.hidden_layers_z[i - 1] @ self.hidden_w[i] + self.hidden_b[i]
#             self.hidden_layers[i] = self.sigmoid(self.hidden_layers_z[i])
#         self.last_layer_z = self.hidden_layers_z[self.depth - 1] @ self.last_w + self.last_b
#         self.last_layer = self.sigmoid(self.last_layer_z)
#         print(self.last_layer)

#     def backpropagate(self, y):
#         last_w_delta = np.zeros((self.output_dim, self.input_dim))
#         last_b_delta = np.zeros((1, self.output_dim))

#         #print("y", y, "| last layer", self.last_layer)
#         #print("LOSS", self.loss(self.last_layer, y))

#         C0_a = self.loss_prime(self.last_layer, y)
#         a_z = self.sigmoid_prime(self.last_layer_z)
#         z_w = self.last_layer

#         delta = np.multiply(C0_a, a_z)
#         last_w_delta = delta @ z_w.T
#         last_b_delta = delta

#         w_delta = np.zeros((self.depth, self.input_dim, self.input_dim))
#         b_delta = np.zeros((self.depth, self.input_dim))
#         current_w = self.last_w
#         current_delta = delta

#         for i in range(self.depth - 1, -1, -1):
#              current_delta = np.multiply(current_delta @ current_w.T, self.sigmoid_prime(self.hidden_layers_z[i]))
#              current_w = self.hidden_w[i].T
#              w_delta[i] = current_delta @ self.hidden_layers[i]
#              b_delta[i] = current_delta[0]

#         return (last_w_delta, last_b_delta, w_delta, b_delta)

#     def globalBackprop(self, X_train, y_train):
#         last_w_delta = np.zeros((self.output_dim, self.input_dim))
#         last_b_delta = np.zeros((1, self.output_dim))

#         w_delta = np.zeros((self.depth, self.input_dim, self.input_dim))
#         b_delta = np.zeros((self.depth, self.input_dim))

#         # M = 1 B = 0
#         B_check = np.where(y_train == 0, 1, 0)
#         M_check = np.where(y_train == 1, 1, 0)
#         expected = np.array([B_check, M_check]).T[:, -1::-1]
#         for i in range(len(y_train)):
#             self.propagate(X_train[i])
#             (lw, lb, hw, hb) = self.backpropagate(expected[:,i])
#             last_w_delta += lw
#             last_b_delta += lb
#             w_delta += hw
#             b_delta += hb
#         num = len(X_train)
#         last_w_delta /= num
#         last_b_delta /= num
#         w_delta /= num
#         b_delta /= num

#         return (last_w_delta, last_b_delta, w_delta, b_delta)

#     def update(self, X_train, y_train):
#         (last_w_delta, last_b_delta, w_delta, b_delta) = self.globalBackprop(X_train, y_train)
#         #print(w_delta, last_w_delta, b_delta, last_b_delta)
#         self.hidden_w -= self.learning_rate * w_delta
#         self.last_w -= self.learning_rate * last_w_delta.T
#         self.hidden_b -= self.learning_rate * b_delta
#         self.last_b -= self.learning_rate * last_b_delta

#     def train(self, X_train, y_train):
#         self.init_xavier()
#         X_train = self.normalise(X_train)
#         for _ in range(self.epochs):
#             self.update(X_train, y_train)

#     def predict(self, X_test):
#         X_test = self.normalise(X_test)
#         y = np.zeros((1, len(X_test)))
#         for i in range(len(X_test)):
#             self.propagate(X_test[i])
#             y[:, i] = self.last_layer[0, 0]
#         return np.where(y > 0.5, 1, 0)

#     def evaluate(self, X_test, y_test):
#         y_pred = self.predict(X_test)
#         precision = np.sum((y_pred == 1) & (y_pred == y_test)) / np.sum(y_pred == 1)
#         recall = np.sum((y_pred == 1) & (y_pred == y_test)) / np.sum(y_test == 1)
#         f1_score = 2 * precision * recall / (precision + recall)
#         return {"precision": precision, "recall": recall, "f1_score": f1_score}


class ClassificationModel:
    def __init__(self, input_dim, output_dim, criterion='gini', max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _normalize_data(self, X):
        X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        return X_norm

    def _calculate_gini(self, y):
        classes = np.unique(y)
        gini = 1
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            gini -= p_cls ** 2
        return gini

    def _calculate_entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            entropy -= p_cls * np.log2(p_cls)
        return entropy

    def _calculate_criterion(self, y):
        if self.criterion == 'gini':
            return self._calculate_gini(y)
        elif self.criterion == 'entropy':
            return self._calculate_entropy(y)
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'entropy'.")

    def _split(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _find_best_split(self, X, y):
        n_features = X.shape[1]
        best_gini = float('inf')
        best_feature_index = None
        best_threshold = None

        for feature_index in range(n_features):
            
            # Check if the feature is continuous or discrete
            if np.all(np.mod(X[:, feature_index], 1) == 0):
                thresholds = np.unique(X[:, feature_index])
            else:
                thresholds = np.linspace(X[:, feature_index].min(), X[:, feature_index].max(), 10)

            for threshold in thresholds:
                _, _, y_left, y_right = self._split(X, y, feature_index, threshold)
                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue
                gini_left = self._calculate_criterion(y_left)
                gini_right = self._calculate_criterion(y_right)
                gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth):
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return np.unique(y)[0]

        best_feature_index, best_threshold = self._find_best_split(X, y)
        if best_feature_index is None:
            return np.unique(y)[0]

        X_left, X_right, y_left, y_right = self._split(X, y, best_feature_index, best_threshold)

        node = {}
        node['feature_index'] = best_feature_index
        node['threshold'] = best_threshold
        node['size'] = len(y)
        node['left'] = self._build_tree(X_left, y_left, depth + 1)
        node['right'] = self._build_tree(X_right, y_right, depth + 1)

        return node


    # def train(self, X_train, y_train): #fit
    #     self.tree_ = self._build_tree(X_train, y_train, 0)

    def train(self, X_train, y_train, X_test=None, y_test=None):

        if X_test is not None and y_test is not None:
            # Tune the min_samples_split parameter
            best_min_samples_split = self.tune_min_samples_split(X_train, y_train, X_test, y_test)

            # Update the model with the best min_samples_split value
            self.min_samples_split = best_min_samples_split

        # Train the model
        self.tree_ = self._build_tree(X_train, y_train, 0)
        # Select the features
        # X_train_selected = self.select_features_by_correlation(X_train, y_train)

        # if X_test is not None:
        #     # Select the same features from the test data
        #     X_test_selected = X_test[:, self.feature_indices_]

        #     if y_test is not None:
        #         # Tune the min_samples_split parameter
        #         best_min_samples_split = self.tune_min_samples_split(X_train_selected, y_train, X_test_selected, y_test)

        #         # Update the model with the best min_samples_split value
        #         self.min_samples_split = best_min_samples_split

        # # Train the model
        # self.tree_ = self._build_tree(X_train_selected, y_train, 0)

    def _predict_instance(self, x, tree):
        if isinstance(tree, dict):
            feature_index = tree['feature_index']
            if x[feature_index] <= tree['threshold']:
                return self._predict_instance(x, tree['left'])
            else:
                return self._predict_instance(x, tree['right'])
        else:
            return tree

    def predict(self, X_test):
        predictions = []
        for x in X_test:
            predictions.append(self._predict_instance(x, self.tree_))
        return np.array(predictions)
    

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        precision = np.sum((y_pred == 1) & (y_pred == y_test)) / np.sum(y_pred == 1)
        recall = np.sum((y_pred == 1) & (y_pred == y_test)) / np.sum(y_test == 1)
        f1_score = 2 * precision * recall / (precision + recall)
        return {"precision": precision, "recall": recall, "f1_score": f1_score}

    def tune_min_samples_split(self, X_train, y_train, X_test, y_test):
        best_precision = 0
        best_min_samples_split = None

        # Define the range of min_samples_split values to try
        min_samples_split_values = range(2, 11)

        for min_samples_split in min_samples_split_values:
            # Temporarily update the min_samples_split value
            self.min_samples_split = min_samples_split

            # Train and evaluate the model
            self.train(X_train, y_train)
            y_pred = self.predict(X_test)
            evaluation = self.evaluate(X_test, y_test)
            precision = evaluation["precision"]

            if precision > best_precision:
                best_precision = precision
                best_min_samples_split = min_samples_split

        # Update the model with the best min_samples_split value
        self.min_samples_split = best_min_samples_split

        print(f"Best min_samples_split: {best_min_samples_split}, Best precision: {best_precision}")
        return best_min_samples_split

    def select_features_by_correlation(self, X, y):
        # Transpose the input data
        X_transposed = X.T

        # Reshape the target variable to a 2D array with a single row
        y_2d = y.reshape(1, -1)

        # Calculate the correlation between each feature and the target variable
        correlations = np.corrcoef(X_transposed, y_2d)[0, :]

        # Filter out features with no variance
        valid_features = np.var(X, axis=0) > 0
        correlations = correlations[valid_features]

        # Sort the features by their correlation
        sorted_feature_indices = np.argsort(correlations)

        # Select the top k features
        k = 5  # You can adjust this value based on your needs
        selected_feature_indices = sorted_feature_indices[-k:]

        # Update the model with the selected features
        self.input_dim = len(selected_feature_indices)
        self.feature_indices_ = selected_feature_indices

        return X[:, selected_feature_indices]
