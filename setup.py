from setuptools import setup, find_packages

setup(
    name='sparcnet',  # Replace with your package's name
    version='0.1.0',    # Version number
    packages=find_packages(),  # Automatically find all packages in the repo
    install_requires=['kneed','torch','torchvision'
    ],
    package_data={
        "sparcnet": ["model_1130.pt"],
    },
    author='HS',
    author_email='haoershi@seas.upenn.edu',
    description='A short description of your package',
    url='https://github.com/haoershi/sparcnet',  # Your GitHub repositor
)
