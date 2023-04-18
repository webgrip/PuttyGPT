FROM python:alpine3.17 AS base


 ENV PACKAGES=

ENV PYTHON_PACKAGES="\
    numpy \
    matplotlib \
    scipy \
    scikit-learn \
    nltk \
"


WORKDIR /app

RUN apk update && \
    apk upgrade

RUN apk add --no-cache --virtual .build-dependencies python3 build-base gcc musl-dev python3-dev openblas-dev libffi-dev openssl-dev g++ gfortran freetype-dev pkgconfig dumb-init musl libc6-compat linux-headers build-base bash git ca-certificates freetype libgfortran libgcc libstdc++ openblas tcl tk
RUN apk add --virtual build-runtime openssh git
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN python3 -m ensurepip
RUN rm -r /usr/lib/python*/ensurepip
RUN pip3 install --upgrade pip setuptools
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf pip3 /usr/bin/pip
RUN rm -r /root/.cache
# RUN pip install --no-cache-dir $PYTHON_PACKAGES
#RUN pip3 install 'pandas<0.21.0' \    #<---------- PANDAS
#RUN apk add --no-cache --virtual build-dependencies 
#RUN apk add --no-cache --virtual build-dependencies gfortran
#RUN apk add --no-cache --virtual build-dependencies gfortran
RUN rm -rf /var/cache/apk/*

#RUN apk update && \
#    apk upgrade && \ 
    #apk add --no-cache --virtual .build-deps gcc musl-dev libffi-dev openssl-dev g++ gfortran libopenblas-dev liblapack-dev pkg-config && \
    #apk add --no-cache --virtual .build-deps gcc g++ gfortran libopenblas-dev liblapack-dev pkg-config && \
    #rm -rf /var/cache/apk/*


# Create a virtual environment and activate it
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Install the Python dependencies


# Copy the requirements file to the container
COPY Eve/requirements.txt .
COPY Eve/requirements-application.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements-application.txt

# Copy the application code to the container
COPY ./Eve ./app

# Set the entrypoint for the container
#CMD [ "python", "main.py" ]
ENTRYPOINT ["tail", "-f", "/dev/null"]