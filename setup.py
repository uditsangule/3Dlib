from setuptools import setup, find_packages
from pathlib import Path
from pkg_resources import parse_requirements

VERSION = '0.2'
DESC = '3D lib for SceneRecon'
LONG_DESC = 'First verion of 3dlib'
with Path('requirements.txt').open() as f: install_req = [str(r).split('~')[0] for r in parse_requirements(f) if
                                                          not str(r).__contains__('-')]

setup(name='lib3d', version=VERSION, author='udit.S', author_email='uditsangule@gmail.com', description=DESC,
      long_description=LONG_DESC, packages=find_packages(), requires=install_req, keywords=['python3', 'find package'],
      classifiers=['Operating System :: Linux'], python_requires='>=3.8')
