#!/bin/bash

function load_compiler(){
	module load gcc/13
	echo "compiler: $(which gcc)"
}


function load_mpi() {
	module load openmpi/5 
	echo "mpicc: $(which mpicc)"
	# Fix in KUMA
	export UCX_NET_DEVICES="mlx5_0:1"
	
}

