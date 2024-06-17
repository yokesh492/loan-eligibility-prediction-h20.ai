FROM ubuntu

RUN apt update -y
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN apt install openjdk-8-jre-headless -y

WORKDIR /app

RUN pip3 install flask pandas h2o

ADD ./output ./output
ADD server.py server.py

EXPOSE 5000

CMD ["python3", "server.py"]