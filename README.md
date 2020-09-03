# BEMPP-CAVITY

An implementation of BEMPP for system with nested boundaries.

Code is available online at https://github.com/sheeshee/MSc-Project-Code

# Installation Instructions

## Prerequisites 

* This project requires Docker. Install it from https://docs.docker.com/engine/install/

## Steps:

* Build the docker image
  ```
  docker build -t cavitybempp:latest .
  ```

* Run the `start.sh` script or the following bash command:
  ```
  docker run -it --rm -v $(pwd):/home/bempp/work -p 8888:8888 cavitybempp:latest
  ```
  This will launch a Jupyter Notebook server. In the console you will see a URL printed. Copy this URL into your web browser (keeping 127.0.0.1 at the beginning of the URL)
