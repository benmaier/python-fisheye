from setuptools import setup

setup(name='fisheye',
      version='0.0.1',
      description="Transform single points or arrays of points using several fisheye functions.",
      url='https://www.github.com/benmaier/python-fisheye',
      author='Benjamin F. Maier',
      author_email='bfmaier@physik.hu-berlin.de',
      license='MIT',
      packages=['fisheye'],
      install_requires=[
          'numpy>=1.14',
          'scipy>=0.17',
      ],
      zip_safe=False)
