from sklearn.mixture import GaussianMixture
from scipy.special import logsumexp
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse


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


mu = np.array([2.0, -1.0])
Sigma = np.array([[2.0, 0.8], [0.8, 1.0]])
gt_gaussian = sample_2d_gaussian(mu, Sigma, 1000)

plt.figure(figsize=(10, 8))
plt.scatter(gt_gaussian[:, 0], gt_gaussian[:, 1], alpha=0.4, s=10)

# plot 1 sigma ellipse


eigenvalues, eigenvectors = np.linalg.eig(Sigma)
angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
ellipse = Ellipse(mu, 2*np.sqrt(eigenvalues[0]), 2*np.sqrt(eigenvalues[1]), 
                  angle=angle, fill=False, color='red', linewidth=2)
plt.gca().add_patch(ellipse)

plt.plot(mu[0], mu[1], 'ro', markersize=8, label='Mean')
plt.xlabel(r"$\chi_1$")
plt.ylabel(r"$\chi_1$")
plt.title('2D Gaussian Samples with 1 sigma Ellipse')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
plt.savefig('gmm_scatter_plot.png')

# MLE estimation
N = gt_gaussian.shape[0]
mu_MLE, Sigma_MLE = MLE_gaussian(gt_gaussian, N)

print("MLE Mean:\n", mu_MLE)
print("MLE Covariance Matrix:\n", Sigma_MLE)
dif_mu = mu - mu_MLE
dif_Sigma = Sigma - Sigma_MLE

# Compute norms
L2_mu = np.linalg.norm(dif_mu, 2)  # Vector L2 norm
norm_Sigma = np.linalg.norm(dif_Sigma, 2)

print("L2 norm of difference in means:", L2_mu)
print("2-norm of difference in covariance matrices:", norm_Sigma)

g1 = GaussianMixture(n_components=1, covariance_type='full', n_init=10, random_state=0).fit(gt_gaussian)
mu_gmm = g1.means_[0]
Sigma_gmm = g1.covariances_[0]
print("GMM Mean:\n", mu_gmm)
print("GMM Covariance Matrix:\n", Sigma_gmm)

diff_mu = mu_MLE - mu_gmm
diff_Sigma = Sigma_MLE - Sigma_gmm

# compute norms
L2_norm_mu = np.linalg.norm(diff_mu, 2)
norm_2_Sigma = np.linalg.norm(diff_Sigma, 2)

print("L2 norm of difference in means:", L2_norm_mu)
print("2-norm of difference in covariance matrices:", norm_2_Sigma)

# no. of new samples to draw from the fitted GMM
M = 500

# Draw M samples from the fitted GMM
new_samples, _ = g1.sample(M)

# Plot original and new samples together
plt.figure(figsize=(10, 8))
plt.scatter(gt_gaussian[:, 0], gt_gaussian[:, 1], alpha=0.4, s=10, label='Original Samples')
plt.scatter(new_samples[:, 0], new_samples[:, 1], color='orange', alpha=0.6, s=20, label='GMM Samples')

# Plot true mean and ellipse for reference
eigenvalues, eigenvectors = np.linalg.eig(Sigma)
angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
ellipse = Ellipse(mu, 2*np.sqrt(eigenvalues[0]), 2*np.sqrt(eigenvalues[1]), 
                  angle=angle, fill=False, color='red', linewidth=2, label='True 1 sigma Ellipse')
plt.gca().add_patch(ellipse)

plt.plot(mu[0], mu[1], 'ro', markersize=8)
plt.xlabel(r'$\chi_1$')
plt.ylabel(r'$\chi_2$')
plt.title(f'Original vs GMM ({M} samples)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()
plt.savefig('gmm_sample_overlay.png')


