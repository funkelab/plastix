from setuptools import setup

setup(
        name='plastix',
        version='0.1',
        description='Differentiable Neural Circuit Simulation',
        url='https://github.com/funkelab/plastix',
        author='Jan Funke',
        author_email='funkej@janelia.hhmi.org',
        license='MIT',
        packages=[
            'plastix',
            'plastix.kernels',
            'plastix.kernels.nodes',
            'plastix.kernels.edges',
            'plastix.layers'
        ],
        install_requires=[
            'jax',
            'numpy',
            'tqdm'
        ]
)
