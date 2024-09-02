#!/usr/bin/bash

# Help message
Help()
{
    echo "Start the Jekyll server for local version of the website."
    echo "If the server cannot be run natively start a docker container"
    echo "and inside it start the server. If the docker image docs"
    echo "does not exist build it first."
    echo
    echo "Syntax: run.sh [-h|-d]"
    echo "options:"
    echo "h     Print this help message."
    echo "d     Force the script to use docker."
}

# Deal with flags
force_docker=false
while getopts ":hd" option; do
    case $option in
        h)  # display Help
            Help
            exit;;
        d)  # force docker
            force_docker=true;;
        \?)  # invalid option
            echo "Error: Invalid option"
            echo
            Help
            exit;;
    esac
done

# If running in docker
run_docker()
{
    # Check if image is available
    if [[ $(docker image inspect docs 2>/dev/null) == "[]" ]]; then
        echo "Building docker image"
        bash ./build.sh  # Build image
    fi
    echo "Running in docker"
    docker run --network="host" -v ./docs:/usr/src/docs/docs -v ./assets:/usr/src/docs/assets -t docs
}

# Main
if $force_docker || ! bundle exec jekyll serve -l -o; then
    run_docker
fi
