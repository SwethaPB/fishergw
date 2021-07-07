## useful links to create a package
## https://python-packaging.readthedocs.io/en/latest/minimal.html#publishing-on-pypi
## https://docs.python-guide.org/writing/structure/#setup-py
## https://docutils.sourceforge.io/docs/user/rst/quickref.html
## https://medium.com/@udiyosovzon/things-you-should-know-when-developing-python-package-5fefc1ea3606

from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='fishergw',
      version='0.0.1',
      description='A Python package to compute Fisher matrices for gravtational wave models',
      long_description=readme(),
      url='https://github.com/costa-pacilio/fishergw.git',
      author='Costantino Pacilio',
      author_email='costantinopacilio1990@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'sympy',
          'scipy',
      ],
      include_package_data=True,
      zip_safe=False)
