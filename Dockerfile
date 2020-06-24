FROM node:alpine
RUN npm i docsify-cli -g
COPY . /data
WORKDIR /data
CMD [ "docsify", "serve", "docs" ]