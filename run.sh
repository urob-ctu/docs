#!/usr/bin/bash
if ! bundle exec jekyll serve -l -o; then
    if [[ $(docker image inspect docs 2>/dev/null) == "[]" ]]; then
        echo "Building docker image"
        bash ./build.sh
    fi
    echo "Running in docker"
    docker run --network="host" -v ./docs:/usr/src/docs/docs -v ./assets:/usr/src/docs/assets -t docs
fi
