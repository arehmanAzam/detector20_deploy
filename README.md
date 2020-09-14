# Deploying detector20 using Flask + Gunicorn + Nginx inside Docker

## Running the solution

In order to run this solution, you just have to install Docker, Docker compose, then clone this repository, and then:
```
bash run_docker.sh
```

For Docker installation instructions follow:

— [Docker installation](https://docs.docker.com/engine/install/ubuntu/)

— [Make Docker run without root](https://docs.docker.com/engine/install/linux-postinstall/)

— [Docker Compose installation](https://docs.docker.com/compose/install/)

## Understanding the solution


— The fast way: the project is structured as follows: Flask app and WSGI entry point are localed in flask_app directory. Nginx and project configuration files are located in nginx directory. Both directories contain Docker files that are connected using docker_compose.yml file in the main directory. 
  
   For simplicity, I also added run_docker.sh file for an even easier setting-up and running this solution. 
```
.
├── flask_app 
│   ├── yolov3_tf2
|      |___Deploy
|          |___Deploy.py
|   |................
|   |................
│   ├── wsgi.py
│   └── Dockerfile (file)
├── nginx
│   ├── nginx.conf          
│   ├── project.conf
│   └── Dockerfile (file
├── docker-compose.yml
└── run_docker.sh
```
### Copy model files to checkpoint directory

-For now the checkpoint files are larger and can't be uploaded in git. Please download it from following link 
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1I9ugHr_dnQD00zMeOKgW26BNqXi6OEwn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1I9ugHr_dnQD00zMeOKgW26BNqXi6OEwn" -O yolov3_train_11.tf.zip && rm -rf /tmp/cookies.txt
```


put it in checkpoints folder

```
->flask_app
-->yolov3_tf2
   |->checkpoints
     |->checkpoint (file)
```

Also rename the model name in checkpoint file accoding to the epoch number and files
For example

For files 

#### yolov3_train_11.tf.data-00000-of-00002, yolov3_train_11.tf.index, yolov3_train_11.tf.data-00001-of-00002

Specify 
#### yolov3_train_11.tf
