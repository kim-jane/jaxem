# jaxem


## Setting up a virtual environment

First, deactivate any active conda environments (if applicable):
`conda deactivate`

Then, Create a new virtual environment and activate:
`python -m venv myenv`
`source myenv/bin/activate`

## Install the dependencies

Simply type:
`pip install -e .`


## Chiral effective field theory potential

This code utilizes the chiral effective field theory potential from:  
A. Gazerlis et al., Phys. Rev. C 90, 054323 (2014).
[https://doi.org/10.1103/PhysRevC.90.054323](https://doi.org/10.1103/PhysRevC.90.054323)  

The Python interface was written by Christian Drischler and is available in the GitHub repository [`cdrischler/greedy_emulator`](https://github.com/cdrischler/greedy_emulator).  
This repository is included as a submodule named `chiral` within this repo.  

The `chiral` submodule does not update automatically. To explicitly update it, run:  

```bash
git submodule update --remote --merge -- chiral
git add chiral
git commit -m "Updated chiral submodule."
git push origin main
```

To compile the necessary Cython interface, do the following:
```bash
cd chiral
export MYLOCAL=${HOME}/src
make
```




