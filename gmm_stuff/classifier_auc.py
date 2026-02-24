from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score
from scipy.spatial.distance import cdist
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc

SEED = 123
rng = np.random.default_rng(SEED)

def sample_2d_gaussian(mu, Sigma, n, rng):
    #ground truth
    if rng is None:
        rng = np.random.default_rng()
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n)  

def MLE_gaussian(X, N):
    mu_MLE = np.mean(X, axis=0)
    center = X - mu_MLE
    Sigma_MLE = (center.T @ center)/ N

    return mu_MLE, Sigma_MLE

def plot_ellipse(mu, Sigma, edgecolor):
    # plot 1 sigma ellipse
    eigenvalues, eigenvectors = np.linalg.eig(Sigma)
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    ellipse = Ellipse(mu, 2*np.sqrt(eigenvalues[0]), 2*np.sqrt(eigenvalues[1]), 
                    angle=angle, fill=False, color=edgecolor, linewidth=2)
    plt.gca().add_patch(ellipse)

def align_labels(true_labels, predicted_labels):
    acc_normal = accuracy_score(true_labels, predicted_labels)
    acc_swapped = accuracy_score(true_labels, 1 - predicted_labels)
    if acc_swapped > acc_normal:
        return 1 - predicted_labels, acc_swapped
    else:
        return predicted_labels, acc_normal

def geom_boundary( X):  
    d = X - m
    dot_products = np.dot(d, r_hat)
    preds = np.where(dot_products < 0, 1, 2)
    return preds

def prob_classifier(prob_class):  
    tau = 0.5
    p = np.where(prob_class < tau, 1, 2)
    return p

w = np.array([0.2, 0.8])
mu_1 = np.array([0.0, 0.0])
Sigma_1 = np.array([[1.0, 0.8], [0.8, 1.5]])

mu_2 = np.array([2.0, 2.0])
Sigma_2 = np.array([[1.2, -0.5], [-0.5, 0.8]])

# sample component indices according to probabilities
rng = np.random.default_rng(SEED)
component_indices = rng.choice([0,1], size=3000, p=w)
N = 3000
samples = np.zeros((N, 2))
for i in range(N):
    if component_indices[i] == 0:
        samples[i, :] = sample_2d_gaussian(mu_1, Sigma_1, 1, rng=rng)
    else:
        samples[i, :] = sample_2d_gaussian(mu_2, Sigma_2, 1, rng=rng)

n_train = int(0.8 * N)
n_test = int(0.2 * N)

X_train = samples[:n_train]
train_components = component_indices[:n_train]

X_test = samples[n_train:]
test_components = component_indices[n_train:]

#Fit k=2 components on X_train

g2 = GaussianMixture(n_components=2, covariance_type='full', n_init=10, random_state=0).fit(X_train)

mu_gmm_2 = g2.means_
Sigma_gmm_2 = g2.covariances_
weights_gmm_2 = g2.weights_
print("GMM Mean:\n", mu_gmm_2)
print("GMM Covariance Matrix:\n", Sigma_gmm_2)
print("GMM Weights:\n", weights_gmm_2)

#Nearest mean rule

d0 = np.linalg.norm(mu_gmm_2[0] - mu_1) + np.linalg.norm(mu_gmm_2[1] - mu_2)
d1 = np.linalg.norm(mu_gmm_2[1] - mu_1) + np.linalg.norm(mu_gmm_2[0] - mu_2)

if d0 <= d1:
    class_to_component = {1: 0, 2: 1}
    component_to_class = {0: 1, 1: 2}
else:
    class_to_component = {1: 1, 2: 0}
    component_to_class = {1: 1, 0: 2}

mu_hat_1 = mu_gmm_2[class_to_component[1]]
mu_hat_2 = mu_gmm_2[class_to_component[2]]
sigma_hat_1 = Sigma_gmm_2[class_to_component[1]]
sigma_hat_2 = Sigma_gmm_2[class_to_component[2]]

print(f"mu_hat_1 = {mu_hat_1}")
print(f"mu_hat_2 = {mu_hat_2}")
print(f"sigma_hat_1 = {sigma_hat_1}")
print(f"sigma_hat_2 = {sigma_hat_2}")

# Verify correctness of GMM model and class-to-component mapping

mu1_error = np.linalg.norm(mu_hat_1 - mu_1) 
mu2_error = np.linalg.norm(mu_hat_2 - mu_2)
sigma1_error = np.linalg.norm(sigma_hat_1 - Sigma_1, 'fro') 
sigma2_error = np.linalg.norm(sigma_hat_2 - Sigma_2, 'fro')

print("Verification of GMM model:\n")
print(f"mu_1 error: {mu1_error:.4f}")
print(f"mu_2 error: {mu2_error:.4f}")
print(f"Sigma_1 error: {sigma1_error:.4f}")
print(f"Sigma_2 error: {sigma2_error:.4f}")

#Training set accu. check

train_predictions_gmm = g2.predict(X_train)
train_predictions_aligned = np.array([component_to_class[c] for c in train_predictions_gmm]) - 1
aligned_preds, train_acc = align_labels(train_components, train_predictions_aligned)
print(f"\n3. Training Accuracy: {train_acc:.4f}")


plt.figure(figsize=(10, 8))

r_hat = (mu_hat_2 - mu_hat_1) / np.linalg.norm(mu_hat_2 - mu_hat_1)
m = (mu_hat_2 + mu_hat_1) / 2

colors = ['red' if c == 0 else 'black' for c in train_components]
plt.scatter(X_train[:, 0], X_train[:, 1], c=colors, alpha=0.3, s=10)

# Plot true parames
plot_ellipse(mu_1, Sigma_1, 'blue')
plot_ellipse(mu_2, Sigma_2, 'yellow')
plt.scatter(*mu_1, color='blue', marker='x', s=200, linewidths=3, label='True mu_1')
plt.scatter(*mu_2, color='yellow', marker='x', s=200, linewidths=3, label='True mu_2')

# Plot estimated params
plot_ellipse(mu_hat_1, sigma_hat_1, 'red')
plot_ellipse(mu_hat_2, sigma_hat_2, 'black')
plt.scatter(*mu_hat_1, color='red', marker='o', s=200, linewidths=3, label='Estimated mu_1_hat')
plt.scatter(*mu_hat_2, color='black', marker='o', s=200, linewidths=3, label='Estimated mu_2_hat')

line_dir = np.array([-r_hat[1], r_hat[0]])
line_points = np.array([m + t * line_dir for t in np.linspace(-5, 5, 100)])

plt.plot(line_points[:, 0], line_points[:, 1], 'g--', linewidth=2, label='Geometric Decision Boundary')

plt.legend()
plt.xlabel(r'$\chi_1$')
plt.ylabel(r'$\chi_2$')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig("verify.png")

#plot with testing points

geom_test_pred = geom_boundary(X_test)
plt.figure(figsize=(10, 8))

colors_test = ['red' if c == 0 else 'black' for c in test_components]
plt.scatter(X_test[:, 0], X_test[:, 1], c=colors_test, alpha=0.3, s=10)

# Plot geometric decision boundary line (same as before)
plt.plot(line_points[:, 0], line_points[:, 1], 'g--', linewidth=2, label='Geometric Decision Boundary')

plt.legend()
plt.title('Testing points and Geometric Decision Boundary')
plt.xlabel(r'$\chi_1$')
plt.ylabel(r'$\chi_2$')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig("teting.png")

geom_test_pred_zero = geom_test_pred - 1
conf_mat = confusion_matrix(test_components, geom_test_pred_zero)
print("Confusion Matrix for Geometric Decision Boundary on Testing Data:")
print(conf_mat)

#Computing P(k|x)
comp_idx_for_class2 = class_to_component[2]
proba_class2 = g2.predict_proba(X_test)[:, comp_idx_for_class2]
prob_class_pred = prob_classifier(proba_class2) -1 
conf_mat2 = confusion_matrix(test_components, prob_class_pred)
print("Confusion Matrix for Probability classification on Testing Data:")
print(conf_mat2)

#roc and auc
fpr, tpr, thresholds = roc_curve(test_components, proba_class2)
auc2 = auc(fpr, tpr)

plt.figure(figsize=(10, 8))

# Plot ROC curve
plt.plot(fpr, tpr, color='darkorange', lw=3, 
         label=f'ROC curve (AUC = {auc2:.4f})')

# Plot diagonal line (random classifier)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random Classifier (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)', fontsize=14, fontweight='bold')
plt.ylabel('True Positive Rate (TPR)', fontsize=14, fontweight='bold')
plt.title('ROC Curve - Probabilistic GMM Classifier\nTest Dataset', 
          fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150, bbox_inches='tight')
print(f"AUC:{auc2}")
plt.show()
plt.savefig("roc_curve.png")
