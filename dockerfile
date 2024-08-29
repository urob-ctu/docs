FROM arm64v8/ruby:3.3

RUN apt-get update && apt-get -y -qq install vim

# throw errors if Gemfile has been modified since Gemfile.lock
RUN bundle config --global frozen 1

WORKDIR /usr/src/docs

COPY Gemfile Gemfile.lock ./

RUN bundle install

COPY . .

COPY run.sh /usr/local/bin/run.sh
RUN chmod +x /usr/local/bin/run.sh

CMD ["run.sh"]
