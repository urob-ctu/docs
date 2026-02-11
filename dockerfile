# Use the official Ruby 3.3 image to match your previous setup
FROM ruby:3.3

# Install essential dependencies for building gems and for Jekyll's JS runtime.
# We've included 'vim' from your old file for convenience.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    nodejs \
    npm \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory. This path is used in the docker-compose.yml volumes.
WORKDIR /usr/src/app

# Copy Gemfiles first to leverage Docker's layer caching.
# Gems will only be re-installed if these files change.
COPY Gemfile* ./

# From your old Dockerfile: excellent practice!
# Ensures bundle install fails if Gemfile.lock is out-of-sync.
RUN bundle config --global frozen 1

# Install the gems
RUN bundle install

# Copy the rest of your application's code into the container.
COPY . .

# Expose the ports for Jekyll and LiveReload.
EXPOSE 4000
EXPOSE 35729

# This is the command that will run when the container starts.
# We've removed --open and added --host=0.0.0.0.
CMD [ "bundle", "exec", "jekyll", "serve", "--host=0.0.0.0", "--livereload" , "--incremental" ]

