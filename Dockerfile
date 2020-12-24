FROM ubuntu:18.04

ENV PATH="/root/.local/bin:${PATH}"
ARG PATH="/root/.local/bin:${PATH}"

EXPOSE 8000

RUN apt update && apt-get install -y htop python3-dev wget git
RUN wget https://bootstrap.pypa.io/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py
COPY . src/
WORKDIR /src/
RUN apt-get install -y gcc
RUN  pip install --user -r requirements.txt
RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
RUN pip install --user -e detectron2_repo
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]