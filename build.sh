#!/usr/bin/bash

# Help message
Help()
{
    echo "Build the docs docker image."
    echo
    echo "Syntax: run.sh [-h]"
    echo "options:"
    echo "h     Print this help message."
}

# Deal with flags
force_docker=false
while getopts ":h" option; do
    case $option in
        h)  # display Help
            Help
            exit;;
        \?)  # invalid option
            echo "Error: Invalid option"
            echo
            Help
            exit;;
    esac
done

# Main
docker build -t docs .
