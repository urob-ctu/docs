#!/usr/bin/bash

# Help message
Help() {
    echo "Build the docs docker image."
    echo
    echo "Syntax: run.sh [-h]"
    echo "options:"
    echo "h     Print this help message."
}

# Process flags
force_docker=false
while getopts ":h" option; do
    case $option in
    h) # display Help
        Help
        exit
        ;;
    \?) # invalid option
        echo "Error: Invalid option"
        echo
        Help
        exit
        ;;
    esac
done

arch=$(uname -m)
if [ $arch == "aarch64" ]; then
    arch="arm64v8"
fi

# Main
docker build -t urob-docs --build-arg ARCH=$arch .
