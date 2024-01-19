FROM python:3.9.12-slim

# Set working directory
ENV WORKDIR=/opt/libface
WORKDIR $WORKDIR

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Install libface from PyPI
RUN pip install libface

COPY . .