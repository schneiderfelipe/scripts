# bash-scripts

A colection of bash scripts.

## Installation

```
cd
git clone git@github.com:schneiderfelipe/bash-scripts.git
```

(If you clone this repository not your home directory, please also run
something like `cd; ln -s /path/to/bash-scripts .`.)

Now add the following lines to your `~/.bashrc` (or, if you use Zsh,
`~/.zshrc`):

```
export BASH_SCRIPTS=~/bash-scripts
source $BASH_SCRIPTS/config_env.sh
```
