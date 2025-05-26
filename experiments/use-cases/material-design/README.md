# Steps to reprdouce the results of ROSE paper for this use case

1- Setup conda env on Polaris from the following docs:
https://radicalpilot.readthedocs.io/en/devel/supported/polaris.html


2- Clone the material design repo:
```sh
git clone https://github.com/pb8294/material-design-bo.git
cd material-design-bo & mkdir rose_polaris & cd rose_polaris
cp * rose_polaris/
```

2- Install ROSE as follows:
```
git clone https://github.com/radical-cybertools/ROSE.git
cd ROSE
pip install .
```

4- Submit the ROSE Job to the resource manager:
```sh
python run_1.py
```

