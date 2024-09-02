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

# Main
docker build -t docs .
