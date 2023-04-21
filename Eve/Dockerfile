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
RUN rm -rf /var/cache/apk/*

RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"



# Install the Python dependencies
RUN pip install --upgrade pip

# Copy the requirements file to the container
COPY Eve/requirements.txt .

RUN /venv/bin/pip install --upgrade pip
RUN /venv/bin/pip install --no-cache-dir -r requirements.txt

COPY ./Eve ./app

#CMD [ "/venv/bin/python", "main.py" ]
ENTRYPOINT ["tail", "-f", "/dev/null"]