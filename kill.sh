#!/usr/bin/bash

# Help message
Help() {
    echo "Find and kill container running the Jekyll server."
    echo "The script assumes only one such container is running"
    echo "as only one server may run at a time. In interactive"
    echo "mode user may choose any running container to kill."
    echo
    echo "Syntax: run.sh [-h|-i]"
    echo "options:"
    echo "h     Print this help message."
    echo "i     Run in interactive mode."
}

# Process flags
interactive=false
while getopts ":hi" option; do
    case $option in
    h) # display Help
        Help
        exit
        ;;
    i) # interactive mode
        interactive=true ;;
    \?) # invalid option
        echo "Error: Invalid option"
        echo
        Help
        exit
        ;;
    esac
done

# Main
containers=$(docker ps)
# Check if there are no containers running (only the header line is present)
if [ $(echo "$containers" | wc -l) -eq 1 ]; then
    echo "No containers found for user"
    exit 1
fi

if $interactive; then
    # Print jobs with line numbers, start from the second line like this [$NUMBER] ...
    echo ""
    echo "Running containers:"

    # Print header with no line number
    echo "$containers" | awk 'NR == 1 {print $0}'
    echo "$containers" | awk 'NR > 1 {print "["NR-1"]", $0}'
    echo ""

    # Prompt the user to select a job
    read -r -p "Enter the number of the container to kill: " container_number

    # Validate user input
    if ! [[ "$container_number" =~ ^[0-9]+$ ]]; then
        echo "Invalid input: Please enter a number."
        exit 1
    fi

    # Extract the job ID corresponding to the selected number
    container_id=$(echo "$containers" | awk 'NR > 1' | awk -v container_number="$container_number" 'NR == container_number {print $1}')
else
    # Extract the container ID corresponding the line with image name "urob-docs"
    container_id=$(echo "$containers" | awk 'NR > 1' | grep urob-docs | grep run.sh | awk 'NR == 1 {print $1}')
fi

# Check if a valid container ID was obtained
if [ -z "$container_id" ]; then
    echo "No valid container found."
    exit 1
fi

docker kill $container_id
