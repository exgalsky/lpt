import camb
import jax.numpy as jnp

#Set up a new set of parameters for CAMB
kmax=10000
pars = camb.CAMBparams()
#This function sets up with one massive neutrino and helium set using BBN consistency
h = 0.68
pars.set_cosmology(H0=h * 100., ombh2=0.049*h**2., omch2=0.261*h**2., mnu=0., omk=0, tau=0.055)     # Using Websky values for validation purposes.
pars.InitPower.set_params(As=2.022e-9, ns=0.965, r=0.03)
pars.set_for_lmax(4000, lens_potential_accuracy=4);
pars.set_matter_power(redshifts=[0], kmax=kmax)
pars.WantTensors = True
pars.max_l_tensor = 3000
pars.max_eta_k_tensor = 10000
results= camb.get_results(pars)

# Copying the details from CAMB_demo
# These are synchronous gauge and normalized to unit primordial curvature perturbation
# The values stored in the array are quantities like Delta_x/k^2, and hence
# are nearly independent of k on large scales. 
# Indices in the transfer_data array are the variable type, the k index, and the redshift index

trans = results.get_matter_transfer_data()
#get kh - the values of k/h at which they are calculated
kh = trans.transfer_data[0,:,0]
#transfer functions for different variables, e.g. CDM density and the Weyl potential
#CDM perturbations have grown, Weyl is O(1) of primordial value on large scales
delta = trans.transfer_data[camb.model.Transfer_cdm-1,:,0]
transfer_data = jnp.asarray([kh, delta]).T



def fetch_transfer():
    return transfer_data.astype(jnp.float32)