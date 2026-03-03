import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---- Classes copied from notebook ----
class XGBoostTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, leaf_weight=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.leaf_weight = leaf_weight
    def is_leaf(self):
        return self.leaf_weight is not None

class XGBoostTree:
    def __init__(self, max_depth=6, min_child_weight=1, reg_lambda=1.0, gamma=0.0):
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.root = None

    def _calculate_leaf_weight(self, gradients, hessians):
        return -np.sum(gradients) / (np.sum(hessians) + self.reg_lambda)

    def _calculate_gain(self, gradients, hessians):
        return 0.5 * (np.sum(gradients) ** 2) / (np.sum(hessians) + self.reg_lambda)

    def _calculate_split_gain(self, left_gradients, left_hessians, right_gradients, right_hessians):
        left_gain = self._calculate_gain(left_gradients, left_hessians)
        right_gain = self._calculate_gain(right_gradients, right_hessians)
        all_gradients = np.concatenate([left_gradients, right_gradients])
        all_hessians = np.concatenate([left_hessians, right_hessians])
        total_gain = self._calculate_gain(all_gradients, all_hessians)
        return left_gain + right_gain - total_gain - self.gamma

    def _find_best_split(self, X, gradients, hessians):
        best_gain = 0
        best_feature = None
        best_threshold = None
        n_samples, n_features = X.shape
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                left_mask = X[:, feature_idx] >= threshold
                right_mask = ~left_mask
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                left_hessian_sum = np.sum(hessians[left_mask])
                right_hessian_sum = np.sum(hessians[right_mask])
                if left_hessian_sum < self.min_child_weight or right_hessian_sum < self.min_child_weight:
                    continue
                gain = self._calculate_split_gain(
                    gradients[left_mask], hessians[left_mask],
                    gradients[right_mask], hessians[right_mask]
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        return best_feature, best_threshold, best_gain

    def _build_tree(self, X, gradients, hessians, current_depth=0):
        n_samples = X.shape[0]
        if current_depth >= self.max_depth or n_samples <= 1:
            leaf_weight = self._calculate_leaf_weight(gradients, hessians)
            return XGBoostTreeNode(leaf_weight=leaf_weight)
        best_feature, best_threshold, best_gain = self._find_best_split(X, gradients, hessians)
        if best_feature is None or best_gain <= 0:
            leaf_weight = self._calculate_leaf_weight(gradients, hessians)
            return XGBoostTreeNode(leaf_weight=leaf_weight)
        left_mask = X[:, best_feature] >= best_threshold
        right_mask = ~left_mask
        left_child = self._build_tree(X[left_mask], gradients[left_mask], hessians[left_mask], current_depth + 1)
        right_child = self._build_tree(X[right_mask], gradients[right_mask], hessians[right_mask], current_depth + 1)
        return XGBoostTreeNode(feature_index=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def fit(self, X, gradients, hessians):
        self.root = self._build_tree(X, gradients, hessians)

    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.leaf_weight
        if x[node.feature_index] >= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

class XGBoostClassifierFromScratch:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 min_child_weight=1, reg_lambda=1.0, gamma=0.0, subsample=1.0,
                 colsample_bytree=1.0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees = []
        self.base_score = 0.5

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _compute_gradients_hessians(self, y_true, y_pred_proba):
        gradients = y_pred_proba - y_true
        hessians = y_pred_proba * (1 - y_pred_proba)
        hessians = np.maximum(hessians, 1e-16)
        return gradients, hessians

    def fit(self, X, y):
        self.trees = []
        n_samples, n_features = X.shape
        self.initial_prediction = np.log(self.base_score / (1 - self.base_score))
        raw_predictions = np.full(n_samples, self.initial_prediction)
        for i in range(self.n_estimators):
            y_pred_proba = self._sigmoid(raw_predictions)
            gradients, hessians = self._compute_gradients_hessians(y, y_pred_proba)
            if self.subsample < 1.0:
                sample_size = int(n_samples * self.subsample)
                indices = np.random.choice(n_samples, size=sample_size, replace=False)
            else:
                indices = np.arange(n_samples)
            if self.colsample_bytree < 1.0:
                n_cols = max(1, int(n_features * self.colsample_bytree))
                col_indices = np.random.choice(n_features, size=n_cols, replace=False)
            else:
                col_indices = np.arange(n_features)
            X_subset = X[np.ix_(indices, col_indices)]
            g_subset = gradients[indices]
            h_subset = hessians[indices]
            tree = XGBoostTree(max_depth=self.max_depth, min_child_weight=self.min_child_weight,
                               reg_lambda=self.reg_lambda, gamma=self.gamma)
            tree.fit(X_subset, g_subset, h_subset)
            self.trees.append((tree, col_indices))
            update = tree.predict(X[:, col_indices])
            raw_predictions += self.learning_rate * update

    def predict_proba(self, X):
        raw_predictions = np.full(X.shape[0], self.initial_prediction)
        for tree, col_indices in self.trees:
            raw_predictions += self.learning_rate * tree.predict(X[:, col_indices])
        return self._sigmoid(raw_predictions)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


# ---- Test with synthetic data ----
print("Generating synthetic dataset...")
X, y = make_classification(n_samples=500, n_features=6, n_informative=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training XGBoost from scratch...")
clf = XGBoostClassifierFromScratch(n_estimators=50, max_depth=3, learning_rate=0.1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {acc:.4f}")

if acc > 0.7:
    print("SUCCESS: XGBoost from scratch is working correctly!")
else:
    print(f"WARNING: Accuracy is {acc:.4f}, which seems low for this dataset.")

print("Test completed.")
