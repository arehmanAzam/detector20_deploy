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
│   └── Dockerfile
├── nginx
│   ├── nginx.conf          
│   ├── project.conf
│   └── Dockerfile
├── docker-compose.yml
└── run_docker.sh
```
### Copy model files to checkpoint directory

-For now the checkpoint files are larger and can't be uploaded in git. Please download it from following link and put it in checkpoints folder
->flask_app
-->yolov3_tf2
---->checkpoints
     |-checkpoint

Also rename the model name in checkpoint file
