import george
import numpy as np
from george.kernels import ExpSquaredKernel
import scipy


var = 2**2
corr_lengths = np.array([1000, 2000])

kernel = var*ExpSquaredKernel(corr_lengths, ndim=2, )

print(kernel.get_parameter_names())
print(kernel.get_parameter_vector())
np.exp(kernel.get_parameter_vector())

top_botm = np.loadtxt("examples/ammer/top_botm.csv",delimiter=",", skiprows=1,usecols=(1,2,3,4))
top = top_botm[:,2]
bottom = top_botm[:,3]
x_y = top_botm[:,0:2]

gp_top = george.GP(kernel, mean=np.mean(top), fit_mean=False,
               white_noise=np.log(0.5**2), fit_white_noise=False)
gp_top.compute(x_y)
print(gp_top.log_likelihood(top))
print(gp_top.grad_log_likelihood(top))

def nll_top(p):
    gp_top.set_parameter_vector(p)
    ll = gp_top.log_likelihood(top, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

def grad_nll_top(p):
    gp_top.set_parameter_vector(p)
    return -gp_top.grad_log_likelihood(top, quiet=True)

gp_top.compute(x_y)

# Print the initial ln-likelihood.
print(gp_top.log_likelihood(top))

# Run the optimization routine.
p0 = gp_top.get_parameter_vector()
results = scipy.optimize.minimize(nll_top, p0, jac=grad_nll_top, method="L-BFGS-B")

# Update the kernel and print the final log-likelihood.
gp_top.set_parameter_vector(results.x)
print(gp_top.log_likelihood(top))

gp_top.get_parameter_names()
params = gp_top.get_parameter_vector()
params = np.concatenate([np.array([params[0]]), np.exp(params[1:])])
params



kernelb = var*ExpSquaredKernel(corr_lengths, ndim=2, )

kernelb.get_parameter_names()
kernelb.get_parameter_vector()


gp_bottom = george.GP(kernel, mean=np.mean(bottom), fit_mean=False,
               white_noise=np.log(0.5**2), fit_white_noise=False)
gp_bottom.compute(x_y)
print(gp_bottom.log_likelihood(bottom))
print(gp_bottom.grad_log_likelihood(bottom))

def nll_bottom(p):
    gp_bottom.set_parameter_vector(p)
    ll = gp_bottom.log_likelihood(bottom, quiet=True)
    return -ll if np.isfinite(ll) else 1e25

def grad_nll_bottom(p):
    gp_bottom.set_parameter_vector(p)
    return -gp_bottom.grad_log_likelihood(bottom, quiet=True)

gp_bottom.compute(x_y)

# Print the initial ln-likelihood.
print(gp_bottom.log_likelihood(bottom))

# Run the optimization routine.
p0 = gp_bottom.get_parameter_vector()
results = scipy.optimize.minimize(nll_bottom, p0, jac=grad_nll_bottom, method="L-BFGS-B", tol = 1e-6)

# Update the kernel and print the final log-likelihood.
gp_bottom.set_parameter_vector(results.x)
print(gp_bottom.log_likelihood(bottom))

gp_bottom.get_parameter_names()
paramsb = gp_bottom.get_parameter_vector()
kern_pars = np.exp(paramsb)
corr_lb = np.sqrt(kern_pars)
corr_lb
paramsb = np.concatenate([np.array([paramsb[0]]), np.exp(paramsb[1:])])
paramsb

