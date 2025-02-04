from setuptools import setup, find_packages

setup(
    name="driftguard",  
    version="0.1.1",  
    author="Kiplangat Korir",
    author_email="korirkiplangat22@gmail.com",
    description="A lightweight Python library for monitoring data and concept drift in machine learning models.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kiplangatkorir/driftguard",  
    packages=find_packages(),  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  
    install_requires=[  
        "numpy",
        "scikit-learn",
        "pandas",
        "scipy",
        "python-dotenv",  
        "matplotlib",  
    ],
    entry_points={  
        'console_scripts': [
            'driftmonitor=driftmonitor.cli:main',
        ],
    },
    include_package_data=True,  
    package_data={
        "": ["*.env"],  
    },
)
