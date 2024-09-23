#!/usr/bin/bash

# Help message
usage() {
    echo "Start the Jekyll server for local version of the website."
    echo "If the server cannot be run natively start a docker container"
    echo "and inside it start the server. If the docker image docs"
    echo "does not exist build it first."
    echo
    echo "Syntax: bash run.sh [options]"
    echo "options:              Every option must be set separately."
    echo "-h, --help            Print this help message."
    echo "-i, --interactive     Docker in interactive mode."
    echo "-b, --build           Force docker image build."
}

PROJECT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
DOCKER_PROJECT_DIR="/usr/src/docs"

# If running in docker
run() {
    # Check if image is available
    if [[ $(docker image inspect urob-docs 2>/dev/null) == "[]" ]]; then
        build
    fi
    echo "Running in docker"
    docker run \
        --network="host" \
        -v "$PROJECT_DIR:$DOCKER_PROJECT_DIR" \
        $docker_mode urob-docs
}

build() {
    echo "Building docker image"
    docker build \
        --build-arg DOCKER_PROJECT_DIR="$DOCKER_PROJECT_DIR" \
        -t urob-docs \
        "$PROJECT_DIR"
}

main() {
    # Process flags
    docker_mode="-td"
    # force_docker=false
    force_build=false
    while [ $# -gt 0 ]; do
        case $1 in
        -h | --help) # display Help
            usage
            exit 0
            ;;
        -i | --interactive) # interactive mode
            docker_mode="-it" ;;
        -b | --build) # force build
            force_build=true ;;
        *) # invalid option
            echo "Error: Invalid option"
            echo
            usage
            exit 1
            ;;
        esac
        shift
    done

    if $force_build; then
        build
    fi
    run
}

set -e
main "$@"
