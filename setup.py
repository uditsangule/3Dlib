from setuptools import setup , find_packages

VERSION = '0.2'
DESC = '3D lib for SceneRecon'
LONG_DESC = 'First verion of 3dlib'

setup(name='lib3d' , version=VERSION,author='udit.S',
      author_email='uditsangule@gmail.com',description=DESC,
      long_description=LONG_DESC,
      packages=find_packages(),
      requires=[],
      keywords=['python3','find package'],
      classifiers=['Operating System :: Linux'],python_requires='>=3.8')
