from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='fishergw',
      version='0.1',
      description='A Python package to compute Fisher matrices for gravtational wave models',
      long_description=readme(),
      url='https://github.com/costa-pacilio/fishergw.git',
      author='Costantino Pacilio',
      author_email='costantinopacilio1990@gmail.com',
      license='MIT',
      packages=['fishergw'],
      install_requires=[
          'sympy',
          'scipy',
      ],
      include_package_data=True,
      zip_safe=False)
