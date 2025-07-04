import sys
from pathlib import Path
sys.path.append(Path(__file__).parent.as_posix())
from setuptools import setup
from build_extension import create_extension, get_build_ext_cmdclass

__version__ = "2.1.0"


setup(
    name="glass",
    version=__version__,
    description="Graph Library for Approximate Similarity Search",
    author="Zihao Wang",
    ext_modules=create_extension(__version__),
    packages=["ann_dataset"],
    cmdclass=get_build_ext_cmdclass(),
    zip_safe=False,
)
