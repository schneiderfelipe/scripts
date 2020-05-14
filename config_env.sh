#!/bin/bash

if [ -z "${SCRIPTS}" ]; then
    SCRIPTS="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

export PATH=${SCRIPTS}:${PATH}
