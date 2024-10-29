#!/bin/bash


export DISAPTCH_CONFIG_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )


HOSTNAME="$(hostname -f)"


if [[ "${HOSTNAME}" == *kuma* || "${HOSTNAME}" == *jed* ]]; then
	source ${DISAPTCH_CONFIG_DIR}/epfl_scitas.sh
	echo "** Enable SCITAS configuration"
else
	source ${DISAPTCH_CONFIG_DIR}/generic.sh
	echo "** Enable Generic configuration"
fi


