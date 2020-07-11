from setuptools import setup, find_packages

with open("README.md", "r") as fh:
	long_desc = fh.read()

setup(name='HEPAutoencoders', 
	version='0.1', 
	description='Tools for training Autoencoders on high energy physics data',	
	long_description=long_desc,
	long_description_content_type="text/markdown",
	url="https://github.com/Autoencoders-compression-anomaly/AE-Compression-pytorch.git",
	author='ATLAS_AECompression_Group',
	packages=find_packages(),
	classifiers=["Programming Language :: Python :: 3.7", "License :: OSI Approved :: Apache Software License", "Operating System :: OS Independent"],
	python_requires='>=3.6')

