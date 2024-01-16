from setuptools import setup
pname='lpt'
setup(name=pname,
      version='0.2',
      description='Massively parallel GPU-enabled Lagrangian perturbation theory in Python using [jax.]numpy',
      url='http://github.com/exgalsky/lpt',
      author='exgalsky collaboration',
      license_files = ('LICENSE',),
      packages=[pname],
      zip_safe=False)
