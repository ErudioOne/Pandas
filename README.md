# Pandas

Pandas training course

## Conda environment

To share the environment, it is used:  

```shell
conda env export --from-history | grep -v "^prefix: " > environment.yml
```

So only packages that are explicitly requested will be included, and without the
prefix, which is the local folder where the package is installed.

After cloning this repository, you can restore the environment using:

```shell
conda env create -f environment.yml
```
