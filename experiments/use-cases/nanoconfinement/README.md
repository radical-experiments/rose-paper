# Steps to reprdouce the results of ROSE paper for this use case

1-setup your modules
```sh
./env.sh
```

2- Clone the main repo and move the files in this folder to the main repo folder:
```sh
git clone https://github.com/softmaterialslab/nanoconfinement-md.git

cp *.py nanoconfinement-md/python/surrogate_samplesize
```

3- Install ROSE as follows:
```
git clone https://github.com/radical-cybertools/ROSE.git
cd ROSE
pip install .
```

4- Submit the ROSE Job to the resource manager:
```sh
./submit_rose.sh
```

