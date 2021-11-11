from setuptools import setup
import io
from os import path

# setup(name='topogenesis',
#       version='0.0.2',
#       description='Topological Structures and Methods for Generative Systems and Sciences',
#       url='https://github.com/shervinazadi/topoGenesis',
#       author='Shervin Azadi, and Pirouz Nourian',
#       author_email='shervinazadi93@gmail.com',
#       license='???',
#       packages=['topogenesis'],
#       zip_safe=False,
#       python_requires='>=3.7',
#       )

here = path.abspath(path.dirname(__file__))

def read(*names, **kwargs):
    return io.open(
        path.join(here, *names),
        encoding=kwargs.get("encoding", "utf8")
    ).read()

long_description = read("README.md")
requirements = read("requirements.txt").split("\n")
optional_requirements = {}

setup(
    name='topogenesis',
    version="0.0.3",
    description="Topological Structures and Methods for Generative Systems and Sciences",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    author="Shervin Azadi, and Pirouz Nourian",
    author_email="shervinazadi93@gmail.com",
    license="MIT license",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    keywords=[],
    project_urls={},
    packages=["topogenesis"],
    package_dir={"": "src"},
    package_data={},
    data_files=[],
    include_package_data=True,
    zip_safe=False,
    install_requires=requirements,
    python_requires=">=3.7",
    extras_require=optional_requirements,
    ext_modules=[],
)