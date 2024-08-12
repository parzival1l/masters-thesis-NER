FROM condaforge/mambaforge
RUN apt update
RUN mkdir /workspace
COPY ./environment.yml ./
# RUN apt-get install gcc python3-dev -y
RUN mamba env create -f environment.yml && mamba clean -afy
RUN echo "source activate charEncoder" > ~/.bashrc

ENV PATH /opt/mamba/envs/env/bin:$PATH

COPY . /workspace
WORKDIR /workspace
EXPOSE 8111
