FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
# FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
# FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime
# FROM nvidia/cuda:10.2-cudnn8-runtime
# ARG PYTHON_VERSION=3.8

# RUN apt-get update && apt-get install -y --no-install-recommends --no-install-suggests \
#           python${PYTHON_VERSION} \
#           python3-pip \
#           python${PYTHON_VERSION}-dev \
# # Change default python
#     && cd /usr/bin \
#     && ln -sf python${PYTHON_VERSION}         python3 \
#     && ln -sf python${PYTHON_VERSION}m        python3m \
#     && ln -sf python${PYTHON_VERSION}-config  python3-config \
#     && ln -sf python${PYTHON_VERSION}m-config python3m-config \
#     && ln -sf python3                         /usr/bin/python \
# # Update pip and add common packages
#     && python -m pip install --upgrade pip \
#     && python -m pip install --upgrade \
#         setuptools \
#         wheel \
#         six \
# # Cleanup
#     && apt-get clean \
#     && rm -rf $HOME/.cache/pip


## DO NOT EDIT the 3 lines.
RUN mkdir /physionet
COPY ./ /physionet
WORKDIR /physionet

# ## Install your dependencies here using apt install, etc.
# RUN apt-get update --fix-missing && apt-get install -y vim git libgomp1 libsndfile1

# ## Include the following line if you have a requirements.txt file.
RUN pip install -r requirements.txt
# RUN pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
