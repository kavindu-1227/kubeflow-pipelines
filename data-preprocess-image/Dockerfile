# Use an official Ubuntu as a parent image
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

#ENV MODEL_FILE_LDA = clf_lda.joblib
#ENV MODEL_FILE_NN = clf_NN.joblib

# Install any needed packages specified in requirements.txt
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt install -y python3.9
    
RUN python3 --version

RUN apt-get -y install python3-pip    

RUN pip3 install --trusted-host pypi.python.org -r requirements.txt

# Expose port 8080 for the app to run on
#EXPOSE 8080

# Run the app
#CMD ["python3", "train.py"]

