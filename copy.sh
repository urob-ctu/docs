#!/usr/bin/bash
containers=$(docker ps)

# Check if there are no containers running (only the header line is present)
if [ $(echo "$containers" | wc -l) -eq 1 ]; then
    echo "No containers found for user"
    exit 1
fi

# Extract the container ID corresponding the line with image name "docs"
container_id=$(echo "$containers" | awk 'NR > 1' | grep docs | awk 'NR == 1 {print $1}')

# Check if a valid container ID was obtained
if [ -z "$container_id" ]; then
    echo "Invalid container ID obtained."
    exit 1
fi

docker cp ./docs "$container_id":/usr/src/docs/
docker cp ./assets "$container_id":/usr/src/docs/
