# Steps to reproduce the results of the ROSE paper for the neutron scattering use case

1- Setup conda env on Polaris from the following docs:
https://radicalpilot.readthedocs.io/en/devel/supported/polaris.html

2- Install ROSE as follows:
```
git clone https://github.com/radical-cybertools/ROSE.git
cd ROSE
pip install .
```

4- Submit the ROSE Job to the resource manager:
```sh
./submit_rose_on_polaris.sh
```

