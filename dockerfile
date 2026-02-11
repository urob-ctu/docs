FROM ruby:3.4

# Install essential dependencies for building gems and for Jekyll's JS runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory. This path is used in the docker-compose.yml volumes.
WORKDIR /usr/src/app

# Copy Gemfile first to leverage Docker's layer caching.
# Gems will only be re-installed if this file changes.
COPY Gemfile ./

# Install the gems
RUN bundle install

# Copy the rest of your application's code into the container.
COPY . .

# Expose the ports for Jekyll and LiveReload.
EXPOSE 4000
EXPOSE 35729

# This is the command that will run when the container starts.
CMD [ "bundle", "exec", "jekyll", "serve", "--host=0.0.0.0", "--livereload" , "--incremental" ]

