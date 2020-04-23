import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='smoothcal',  
     version='0.0.1',
     scripts=[] ,
     author="Landman Bester",
     author_email="lbester@ska.ac.za",
     description="Calibration using smooth gain regularisaion",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/landmanbester/SmoothCal",
     packages=setuptools.find_packages(),
     install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'numba',
          'astropy==3.2.1',
          'python-casacore==3.0.0',
          'dask',
          "dask[array]",
          "dask-ms[xarray]",
          "sympy",
      ],
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )