# SWAG-Diagonal

[A simple baseline for bayesian uncertainty in deep learning](https://arxiv.org/abs/1902.02476) suggests the bayesian inference using stochastic weight averaging(SWA) with Gaussian-modeled weights(SWAG).



# Uncertainty of unstable weight

    
![png](SWAG_weight_experiments_files/SWAG_weight_experiments_4_0.png)
    
![png](SWAG_weight_experiments_files/SWAG_weight_experiments_5_0.png)
    


# Uncertainty of converged weight
    
![png](SWAG_weight_experiments_files/SWAG_weight_experiments_9_0.png)
    
plt.savefig('var_converge.png', dpi=300)
```


    
![png](SWAG_weight_experiments_files/SWAG_weight_experiments_10_0.png)
    


# SWAG full covariance with 2-dim weights

Although the diagonal covariance approximation is standard in Bayesian deep learning, the author suggest the full-covariance matrix since the diagonal can be too restrictive.

## The full covariance matrix

$$\Sigma = \frac{1}{T-1} \sum^T_{i=1}(\theta_i - \theta_{SWA})(\theta_i - \theta_{SWA})^T$$
Where T is the number of epochs

But we cannot access into $\theta_{SWA}$ during the epoch since the $\theta_{SWA}$ can be calculated in the T'th epoch. So we approximate as

$$\Sigma = \frac{1}{T-1} \sum^T_{i=1}(\theta_i - \bar{\theta_i})(\theta_i - \bar{\theta_i})^T$$
Where $\bar{\theta_i}$ is the running estimate of the parameters' mean obtained from the first $i$ samples.

## The final distribution for sampling weights

$$\mathcal{N}(\theta_{SWA}, (\Sigma_{diag} + \Sigma_{full})/2)$$

## Low-rank covariance
Authors also suggest low-rank covariance that only take the last $K$ observations of weights,
$$\Sigma_{lowrank} = \frac{1}{K} \sum_{[T-K:T]}(\theta_i - \bar{\theta_i})(\theta_i - \bar{\theta_i})^T$$


```python
theta1 = np.random.randn(length) + np.array(list(map(lambda x:x*(x-20), range(-length//2,length//2))))/length
theta2 = np.random.randn(length)/np.arange(1,length+1) #+ np.arange(length)/100

params = np.concatenate((np.expand_dims(theta1,0), np.expand_dims(theta2,0)), axis=0).transpose()

theta1_swa = []
theta2_swa = []

theta1_bar_square = []
theta2_bar_square = []

for i in range(len(theta)):
    sum_theta1 = 0
    sum_theta2 = 0
    sum_theta1_square = 0
    sum_theta2_square = 0
    for j in range(i+1):
        sum_theta1 += theta1[j]
        sum_theta2 += theta2[j]
        sum_theta1_square += theta1[j] * theta1[j]
        sum_theta2_square += theta2[j] * theta2[j]
    theta1_swa.append(sum_theta1/(i+1))
    theta1_bar_square.append(sum_theta1_square/(i+1))
    theta2_swa.append(sum_theta2/(i+1))
    theta2_bar_square.append(sum_theta2_square/(i+1))

### Calculate digonal cov
theta1_bar_square = np.array(theta1_bar_square)
theta1_swa_square = np.array(list(map(lambda x:x*x, theta1_swa)))
var1 = np.sqrt(np.abs(theta1_bar_square - theta1_swa_square))

theta2_bar_square = np.array(theta2_bar_square)
theta2_swa_square = np.array(list(map(lambda x:x*x, theta2_swa)))
var2 = np.sqrt(np.abs(theta2_bar_square - theta2_swa_square))

cov_diag_per_epoch = np.array(list(map(np.diag,np.concatenate(([var1],[var2])).transpose())))

### Calculate Full cov
params_swa = np.concatenate(([theta1_swa], [theta2_swa]), axis=0).transpose()
cov_full = np.zeros((2,2))
cov_full_per_epoch = []
cov_full_per_epoch.append(cov_full)
t = 1
for p_i, p_swa_i in zip(params,params_swa):
    D_i = p_i - p_swa_i
    cov_full += D_i.reshape(2,1)@D_i.reshape(1,2)
    if t >= 2:
        cov_full_per_epoch.append(cov/t)
    t += 1
cov_full_per_epoch = np.array(cov_full_per_epoch)

### Full cov + Diag cov
cov_per_epoch = cov_full_per_epoch/2 + cov_diag_per_epoch/2

fig = plt.figure(figsize=(4,4))
plt.matshow(cov_full_per_epoch[-1])
plt.colorbar()
plt.matshow(cov_diag_per_epoch[-1])
plt.colorbar()
plt.matshow(cov_per_epoch[-1])
plt.colorbar()
plt.show()
```


    <Figure size 288x288 with 0 Axes>



    
![png](SWAG_weight_experiments_files/SWAG_weight_experiments_12_1.png)
    



    
![png](SWAG_weight_experiments_files/SWAG_weight_experiments_12_2.png)
    



    
![png](SWAG_weight_experiments_files/SWAG_weight_experiments_12_3.png)
    



```python
fig = plt.figure(figsize=(14,8))
plt.plot(theta1)
plt.plot(theta2)
plt.xlabel("SGD epochs", fontsize=20)
plt.ylabel("weight value", fontsize=20)
plt.legend(["theta1", "theta2"], fontsize=20)
plt.show()
```


    
![png](SWAG_weight_experiments_files/SWAG_weight_experiments_13_0.png)
    


# Bayesian inference with SWAG weights

## MAP optimization

We can maximize log posterior:
$$log p(\theta|\mathcal{D}) = log p(\mathcal{D}|\theta) + log p(\theta)$$
We can consider the prior $p(\theta)$ as the regularizer in optimization (L1, L2). **But it is not Bayesian inference** since the maximum posterior $\hat{\theta}_{MAP}$ is **determined** so we cannot get non-deterministic outputs (without uncertainty predictions).

## Marginalized predictive distribution
So we marginalizes the posterior distribution over all possible $\theta$.
$$p(y|\mathcal{D},x) = \int p(y|\theta,x) p(\theta|\mathcal{D})$$
In practice, we can approximate this intractable value with Monte carlo sampling.
$$p(y|\mathcal{D},x) \approx \frac{1}{T} \sum_{t} p(y|\theta_t,x)$$
$$\theta_t \sim p(\theta|\mathcal{D})$$

**We can extract the prediction uncertainty by calculating empirical variance of $p(y|\mathcal{D},x)$**


```python

```
