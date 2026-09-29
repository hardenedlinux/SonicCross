# SonicCross

SonicCross is a pure Guile 3.0 / Scheme project. This repository currently
contains only the project infrastructure and build scaffolding.

## Build

```sh
./bootstrap
./configure
make
make check
```

For source-tree development, use the generated `pre-inst-env` wrapper:

```sh
./pre-inst-env guild torchgen --help
```

After installation, the command is discovered by Guile's standard `guild`
dispatcher as the `(scripts torchgen)` module:

```sh
guild torchgen --help
```

The semantic compiler and Core IR are intentionally not implemented yet.
