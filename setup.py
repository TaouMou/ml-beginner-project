from setuptools import setup, find_packages
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirement packages.
    """
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [requirement.strip() for requirement in requirements if requirement.strip()]

        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

setup(
    name="ml_project",
    version="0.1",
    author="TaouMou",
    author_email="mouhcinetaoufiq@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)