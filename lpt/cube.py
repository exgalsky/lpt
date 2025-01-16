import jax
import sys
import os
import gc

import xgmutil as mu

import jax.numpy as jnp 
import jax.random as rnd

class Cube:
    '''Cube'''
    def __init__(self, **kwargs):

        self.N       = kwargs.get('N',512)
        self.Lbox    = kwargs.get('Lbox',7700.0)
        self.partype = kwargs.get('partype','jaxshard')
        self.nlpt    = kwargs.get('nlpt',2)

        self.dk  = 2*jnp.pi/self.Lbox
        self.d3k = self.dk * self.dk * self.dk

        self.s1lpt = None
        self.s2lpt = None

        self.rshape       = (self.N,self.N,self.N)
        self.cshape       = (self.N,self.N,self.N//2+1)
        self.rshape_local = (self.N,self.N,self.N)
        self.cshape_local = (self.N,self.N,self.N//2+1)

        self.start = 0
        self.end   = self.N

        # needed for running on CPU with a signle process
        self.ngpus   = 1        
        self.host_id = 0

        if self.partype == 'jaxshard':
            self.ngpus   = int(os.environ.get("XGSMENV_NGPUS"))
            self.host_id = jax.process_index()
            self.start   = self.host_id * self.N // self.ngpus
            self.end     = (self.host_id + 1) * self.N // self.ngpus
            self.rshape_local = (self.N, self.N // self.ngpus, self.N)
            self.cshape_local = (self.N, self.N // self.ngpus, self.N // 2 + 1)

    def k_axis(self, r=False, slab_axis=False):
        if r: 
            k_i = (jnp.fft.rfftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
        else:
            k_i = (jnp.fft.fftfreq(self.N) * self.dk * self.N).astype(jnp.float32)
        if slab_axis: return (k_i[self.start:self.end]).astype(jnp.float32)
        return k_i
    
    def k_square(self, kx, ky, kz):
        kxa,kya,kza = jnp.meshgrid(kx,ky,kz,indexing='ij')
        del kx, ky, kz ; gc.collect()

        k2 = (kxa**2+kya**2+kza**2).astype(jnp.float32)
        del kxa, kya, kza ; gc.collect()

        return k2
    
    def interp2kgrid(self, k_1d, f_1d):
        kx = self.k_axis()
        ky = self.k_axis(slab_axis=True)
        kz = self.k_axis(r=True)

        interp_fcn = jnp.sqrt(self.k_square(kx, ky, kz)).ravel()
        del kx, ky, kz ; gc.collect()

        interp_fcn = jnp.interp(interp_fcn, k_1d, f_1d, left='extrapolate', right='extrapolate')
        return jnp.reshape(interp_fcn, self.cshape_local).astype(jnp.float32)

    def _generate_sharded_noise(self, N, noisetype, seed, nsub):           
        ngpus   = self.ngpus
        host_id = self.host_id
        start   = self.start
        end     = self.end

        stream = mu.Stream(seedkey=seed,nsub=nsub)
        noise = stream.generate(start=start*N**2,size=(end-start)*N**2).astype(jnp.float32)
        noise = jnp.reshape(noise,(end-start,N,N))
        return jnp.transpose(noise,(1,0,2)) 

    def _generate_serial_noise(self, N, noisetype, seed, nsub):
        stream = mu.Stream(seedkey=seed,nsub=nsub)
        noise = stream.generate(start=0,size=N**3).astype(jnp.float32)
        noise = jnp.reshape(noise,(N,N,N))
        return jnp.transpose(noise,(1,0,2))

    def _apply_grid_transfer_function(self, field, transfer_data):
        transfer_cdm = self.interp2kgrid(transfer_data[0], transfer_data[1])
        del transfer_data ; gc.collect()

        return field*transfer_cdm

    def _fft(self,x_np,direction='r2c'):
        
        from . import multihost_rfft
        from jax import jit
        from jax.experimental import mesh_utils
        from jax.experimental.multihost_utils import sync_global_devices
        from jax.sharding import Mesh, PartitionSpec as P, NamedSharding 
        
        num_gpus = self.ngpus
        if direction=='r2c':
            global_shape = self.rshape
        else:
            global_shape = self.cshape

        devices = mesh_utils.create_device_mesh((num_gpus,))
        mesh = Mesh(devices, axis_names=('gpus',))
        with mesh:
            x_single = jax.device_put(x_np).block_until_ready()
            del x_np ; gc.collect()
            xshard = jax.make_array_from_single_device_arrays(
                global_shape,
                NamedSharding(mesh, P(None, "gpus")),
                [x_single]).block_until_ready()
            del x_single ; gc.collect()
            if direction=='r2c':
                rfftn_jit = jit(
                    multihost_rfft.rfftn,
                    in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                    out_shardings=(NamedSharding(mesh, P(None, "gpus")))
                )
            else:
                irfftn_jit = jit(
                    multihost_rfft.irfftn,
                    in_shardings=(NamedSharding(mesh, P(None, "gpus"))),
                    out_shardings=(NamedSharding(mesh, P(None, "gpus")))
                )
            sync_global_devices("wait for compiler output")

            with jax.spmd_mode('allow_all'):

                if direction=='r2c':
                    out_jit: jax.Array = rfftn_jit(xshard).block_until_ready()
                else:
                    out_jit: jax.Array = irfftn_jit(xshard).block_until_ready()
                sync_global_devices("loop")
                local_out_subset = out_jit.addressable_data(0)
        return local_out_subset

    def generate_noise(self, noisetype='white', nsub=1024**3, seed=13579):

        N = self.N

        noise = None
        if self.partype is None:
            noise = self._generate_serial_noise(N, noisetype, seed, nsub)
        elif self.partype == 'jaxshard':
            noise = self._generate_sharded_noise(N, noisetype, seed, nsub)
        return noise

    def noise2delta(self, delta, power):
        import numpy as np
        if not isinstance(power, (np.ndarray, jnp.ndarray)):
            power = power()
        transfer = power
        p_whitenoise = (2*np.pi)**3/(self.d3k*self.N**3) # white noise power spectrum
        transfer[1] = (power[1] / p_whitenoise)**0.5 # transfer(k) = sqrt[P(k)/P_whitenoise]
        if transfer.ndim != 2 : print('ERROR: Transfer function ndarray is not two dimensional')
        if transfer.shape[0] != 2: print('ERROR: Transfer function ndarray is a two column array. More than two columns supplied.')
        transfer = jnp.asarray(transfer)

        return self._fft(
                    self._apply_grid_transfer_function(self._fft(delta), transfer),
                    direction='c2r')

    def slpt(self, infield='noise', delta=None, mode='lean'):

        if self.nlpt <= 0: return

        kx = self.k_axis()
        ky = self.k_axis(slab_axis=True)
        kz = self.k_axis(r=True)

        k2 = self.k_square(kx, ky, kz)
        
        kx = kx.at[self.N//2].set(0.0)
        kz = kz.at[-1].set(0.0)

        if self.start <= self.N//2 and self.end > self.N//2:
            ky = ky.at[self.N//2-self.start].set(0.0)

        kx = kx[:,None,None]
        ky = ky[None,:,None]
        kz = kz[None,None,:]

        index0 = jnp.nonzero(k2==0.0)

        def _get_shear_factor(ki,kj,delta):
            arr = ki*kj/k2*delta
            if self.host_id == 0: 
                arr = arr.at[index0].set(0.0+0.0j)
            return self._fft(arr,direction='c2r')

        def _delta_to_s(ki,delta):
            # convention:
            #   Y_k = Sum_j=0^n-1 [ X_j * e^(- 2pi * sqrt(-1) * j * k / n)]
            # where
            #   Y_k is complex transform of real X_j
            arr = (0+1j)*ki/k2*delta
            if self.host_id == 0: 
                arr = arr.at[index0].set(0.0+0.0j)
            arr = self._fft(arr,direction='c2r')
            return arr

        if infield == 'noise':
            # FT of delta from noise
            delta = self._apply_grid_transfer_function(self._fft(delta))
        elif infield == 'delta':
            # FT of delta
            delta = self._fft(delta)
        else:
            import numpy as np
            # delta from external file
            delta = jnp.asarray(np.fromfile(infield,dtype=jnp.float32,count=self.N*self.N*self.N))
            delta = jnp.reshape(delta,self.rshape)
            delta = delta[:,self.start:self.end,:]
            # FT of delta
            delta = self._fft(delta)
    
        # Definitions used for LPT
        #   grad.S^(n) = - delta^(n)
        # where
        #   delta^(1) = linear density contrast
        #   delta^(2) = Sum [ dSi/dqi * dSj/dqj - (dSi/dqj)^2]
        #   x(q) = q + D * S^(1) + f * D^2 * S^(2)
        # with
        #   f = + 3/7 Omegam_m^(-1/143)
        # being a good approximation for a flat universe

        if mode == 'fast' and self.nlpt > 1:
            # minimize operations
            sxx = _get_shear_factor(kx,kx,delta)
            syy = _get_shear_factor(ky,ky,delta)
            delta2  = sxx * syy

            szz = _get_shear_factor(kz,kz,delta)
            delta2 += sxx * szz ; del sxx; gc.collect()
            delta2 += syy * szz ; del syy ; del szz; gc.collect()

            sxy = _get_shear_factor(kx,ky,delta)
            delta2 -= sxy * sxy ; del sxy; gc.collect()

            sxz = _get_shear_factor(kx,kz,delta)
            delta2 -= sxz * sxz ; del sxz; gc.collect()

            syz = _get_shear_factor(ky,kz,delta)
            delta2 -= syz * syz ; del syz; gc.collect()

            delta2 = self._fft(delta2)

        elif self.nlpt > 1:
            # minimize memory footprint
            delta2  = self._fft(
                    _get_shear_factor(kx,kx,delta)*_get_shear_factor(ky,ky,delta)
                  + _get_shear_factor(kx,kx,delta)*_get_shear_factor(kz,kz,delta)
                  + _get_shear_factor(ky,ky,delta)*_get_shear_factor(kz,kz,delta)
                  - _get_shear_factor(kx,ky,delta)*_get_shear_factor(kx,ky,delta)
                  - _get_shear_factor(kx,kz,delta)*_get_shear_factor(kx,kz,delta)
                  - _get_shear_factor(ky,kz,delta)*_get_shear_factor(ky,kz,delta))

        if self.nlpt > 1:
            # 2nd order displacements
            self.s2x = _delta_to_s(kx,delta2)
            self.s2y = _delta_to_s(ky,delta2)
            self.s2z = _delta_to_s(kz,delta2)

            del delta2; gc.collect()

        # 1st order displacements
        self.s1x = _delta_to_s(kx,delta)
        self.s1y = _delta_to_s(ky,delta)
        self.s1z = _delta_to_s(kz,delta)

        del delta; gc.collect()


    

