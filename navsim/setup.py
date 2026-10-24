import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Installs
setuptools.setup(
    name="navsim",
    version="1.1.0",
    author="University of Tuebingen",
    author_email="kashyap.chitta@uni-tuebingen.de",
    description="NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking",
    url="https://github.com/autonomousvision/navsim",
    python_requires=">=3.9",
    packages=setuptools.find_packages(script_folder),
    package_dir={"": "."},
    package_data={"navsim.agents.actworld_jepa.inference": ["fast_hist.c"]},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: Apache Software License",
    ],
    license="apache-2.0",
)
