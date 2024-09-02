#!/usr/bin/bash
if ! bundle exec jekyll serve -l -o
then
    docker run --network="host" -v ./docs:/usr/src/docs/docs -v ./assets:/usr/src/docs/assets -t docs
fi
