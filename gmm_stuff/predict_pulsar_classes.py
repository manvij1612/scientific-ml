from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import accuracy_score, confusion_matrix, adjusted_rand_score

def sample_2d_gaussian(mu, Sigma, n, rng=None):
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

w = np.array([0.2, 0.8])
mu_1 = np.array([0.0, 0.0])
Sigma_1 = np.array([[1.0, 0.6], [0.6, 1.5]])

mu_2 = np.array([4.0, 3.0])
Sigma_2 = np.array([[1.2, -0.5], [-0.5, 0.8]])

# sample component indices according to probabilities
rng = np.random.default_rng()
component_indices = rng.choice([0,1], size=1000, p=w)

samples = np.zeros((1000, 2))
for i in range(1000):
    if component_indices[i] == 0:
        samples[i, :] = sample_2d_gaussian(mu_1, Sigma_1, 1, rng=rng)
    else:
        samples[i, :] = sample_2d_gaussian(mu_2, Sigma_2, 1, rng=rng)

plt.figure(figsize=(10, 8))
plt.scatter(samples[component_indices==0, 0], samples[component_indices==0, 1], 
            color='red', alpha=0.6, s=15, label='Type 1 (w=0.2)')
plt.scatter(samples[component_indices==1, 0], samples[component_indices==1, 1], 
            color='black', alpha=0.6, s=15, label='Type 2 (w=0.8)')

plot_ellipse(mu_1, Sigma_1, 'red')
plot_ellipse(mu_2, Sigma_2, 'black')

plt.xlabel(r"$x_1$ (Period $P$)")
plt.ylabel(r"$x_2$ (Spin-down $\dot{P}$)")
plt.title("Samples from Ground Truth Pulsar Gaussian Mixture")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
plt.savefig('gmm_plot.png')

g1 = GaussianMixture(n_components=1, covariance_type='full', n_init=10, random_state=0).fit(samples)
g2 = GaussianMixture(n_components=2, covariance_type='full', n_init=10, random_state=0).fit(samples)

N = len(samples)
X = samples

# Log-likelihood
logL1 = g1.score(X) * N
logL2 = g2.score(X) * N

# Bayesian Information Criterion
bic1 = g1.bic(X)
bic2 = g2.bic(X)

print("GMM with 1 component: Log-Likelihood =", logL1, ", BIC =", bic1)
print("GMM with 2 components: Log-Likelihood =", logL2, ", BIC =", bic2)

mu_gmm_2 = g2.means_
Sigma_gmm_2 = g2.covariances_
weights_gmm_2 = g2.weights_
print("GMM Mean:\n", mu_gmm_2)
print("GMM Covariance Matrix:\n", Sigma_gmm_2)
print("GMM Weights:\n", weights_gmm_2)

# Sampling from fitted model

M = 500
new_samples, _ = g2.sample(M)

# Plot original and new samples together
plt.figure(figsize=(10, 8))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.4, s=10, label='Original Samples')
plt.scatter(new_samples[:, 0], new_samples[:, 1], color='orange', alpha=0.6, s=20, label='GMM Samples')
plt.xlabel(r"$x_1$ (Period $P$)")
plt.ylabel(r"$x_2$ (Spin-down $\dot{P}$)")
plt.title("True samples vs old samples from GMM (M=500)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
plt.savefig('gmm_plot_2.png')

# Predict pulsar classes using the fitted GMM model
predicted_labels = g2.predict(samples)

aligned_predicted, fixed_accuracy = align_labels(component_indices, predicted_labels)

conf_matrix_fixed = confusion_matrix(component_indices, aligned_predicted)
ari = adjusted_rand_score(component_indices, aligned_predicted)

print("Accuracy:", fixed_accuracy)
print("Adjusted Rand Index:", ari)
print("Confusion Matrix:\n", conf_matrix_fixed)

# Plot results
plt.figure(figsize=(10, 8))
plt.scatter(samples[aligned_predicted == 0, 0], samples[aligned_predicted == 0, 1],
            color='blue', s=15, alpha=0.6, label='Predicted Type 1')
plt.scatter(samples[aligned_predicted == 1, 0], samples[aligned_predicted == 1, 1],
            color='green', s=15, alpha=0.6, label='Predicted Type 2')
plt.xlabel(r"$x_1$ (Period $P$)")
plt.ylabel(r"$x_2$ (Spin-down $\dot{P}$)")
plt.title("Predicted Pulsar Classes with GMM")
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
plt.savefig('gmm_plot_3.png')

geom_test_pred_zero_based = geom_test_pred - 1
conf_mat = confusion_matrix(test_components, geom_test_pred_zero_based)
print("Confusion Matrix for Geometric Decision Boundary on Testing Data:")
print(conf_mat)
