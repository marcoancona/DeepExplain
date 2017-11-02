from setuptools import setup

setup(name='deepexplain',
      version='0.1',
      description='Perturbation and gradient-based methods for Deep Network interpretability',
      url='https://github.com/marcoancona/DeepExplain',
      author='Marco Ancona (ETH Zurich)',
      author_email='marco.ancona@inf.ethz.ch',
      license='MIT',
      packages=['deepexplain'],
      install_requires=[
          'numpy>=1.12.1',
          'termcolor'
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      )