FROM ruby:3.3

ARG DOCKER_PROJECT_DIR

RUN apt-get update && apt-get -y -qq install vim

# Copy the Gemfile and Gemfile.lock into the image and run bundle install
RUN mkdir -p $DOCKER_PROJECT_DIR
COPY Gemfile Gemfile.lock $DOCKER_PROJECT_DIR/

WORKDIR $DOCKER_PROJECT_DIR

# throw errors if Gemfile has been modified since Gemfile.lock
RUN bundle config --global frozen 1
RUN bundle install

CMD ["bash", "-c", "bundle exec jekyll serve --livereload --open"]
