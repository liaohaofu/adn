FROM nvidia/cuda:9.0-base

RUN apt update && apt install -y wget unzip curl bzip2 git default-jre libxmu6
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda install pytorch torchvision cudatoolkit=9.0 -c pytorch

RUN git clone https://github.com/liaohaofu/adn.git && cd adn && pip install -r requirements.txt

WORKDIR /adn
