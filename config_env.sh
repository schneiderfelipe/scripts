#!/bin/bash

if [ -z "${BASH_SCRIPTS}" ]; then
    BASH_SCRIPTS="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export PATH=${BASH_SCRIPTS}:${PATH}
