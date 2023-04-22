FROM python:alpine3.17 AS builder

RUN apk update && \
    apk upgrade

RUN apk add --no-cache --virtual .build-dependencies python3 py3-pip build-base gcc musl-dev python3-dev openblas-dev libffi-dev openssl-dev g++ gfortran freetype-dev pkgconfig dumb-init musl libc6-compat linux-headers build-base bash git ca-certificates freetype libgfortran libgcc libstdc++ openblas tcl tk
RUN apk add --virtual build-runtime openssh git
RUN ln -s /usr/include/locale.h /usr/include/xlocale.h
RUN pip3 install --upgrade pip setuptools
RUN ln -sf /usr/bin/python3 /usr/bin/python
RUN ln -sf pip3 /usr/bin/pip
RUN rm -r /root/.cache
RUN rm -rf /var/cache/apk/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY Eve/requirements.txt .

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

FROM builder AS final

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /usr/include/xlocale.h /usr/include/xlocale.h
ENV PATH=/opt/venv/bin:$PATH

WORKDIR /app

COPY ./Eve ./app

#CMD [ "/venv/bin/python", "main.py" ]
ENTRYPOINT ["tail", "-f", "/dev/null"]