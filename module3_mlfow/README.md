# Mlflow

### Module Description

### Repo Structure

### Running the Project

#### Step 1. Setting up Mlflow Server

1. Navigate to module 3 folder on your filesystem
```
cd <project_diretory>/module3_mlflow
```
2. Build and image for an mlflow server container
```
docker build -t mlflow_server -f Dockerfile_server .
```
3. Start a server container in a detached mode
```
docker run -d -p 5001:5000 -v server_storage:/server --name server mlflow_server
```
Note that all the data will be stored in a named volume `server_storage` and will be accessible only through the mlflow server. (I guess we can link another container to that storage if that's really needed.)

4. Check that the server is running
```
docker logs server
```
Don't click on that url from the logs. It won't work, since we've bound port 5001 on localhost to a port 5000 in a container. Instead open mlflow ui in a browser at http://0.0.0.0:5001/ .

#### Step 2. Setting up Mlflow Client/Project Container

5. Build a client image
```
docker build -t mlflow_client .
```
6. Run the client container
```
docker run -it -v ./:/code --name client -p 5002:5002 mlflow_client
```
Note: sometimes this container for some reason eats half of my RAM. Then restarting and reattaching to it helps with this issue.

#### Step 3. Running Mlflow Project

7. From the host console get IP adress of the server container in `docker inspect server` (I really don't want to write a bash script to autoinsert it :( ). In Linux we can direcly grab it by 
```
docker inspect server | grep -Po '(?<="IPAddress": ")[^"]*'
```
8. Set up environmental variable in the *client* container to
```
export MLFLOW_TRACKING_URI=http://<server_IP>:5000/
```
For example, `export MLFLOW_TRACKING_URI=http://172.17.0.2:5000/`. Note that both containers are inside one docker network, and from inside that network server is at port 5000. ( Whereas from the host it's at http://0.0.0.0:5001 ). Run something like `mlflow experiments search` to check that the connection is established properly.

9. Model will be automatically registered as Version 1
```
mlflow run -e linear_regression --env-manager local --experiment-name Basic_Models --run-name lr .
```
10. Remove scaler from trees
```
mlflow run -e boosting --env-manager local --experiment-name Basic_Models --run-name hbr .
```
11. 
```
mlflow run -e forest --env-manager local --experiment-name Basic_Models --run-name rfr .
```
12. 




export MLFLOW_TRACKING_URI=http://localhost:5001/

mlflow run -e linear_regression --env-manager local --experiment-name Basic_Models --run-name lr .

mlflow run -e tuning --env-manager local --experiment-name tuning_cont --run-name hpopt .




mlflow models serve --model-uri runs:/b6f7fcb6047549a6b6378d6e55035064/LinearRegression --no-conda -p 5002


# works from the host. Try in container
mlflow models serve --model-uri runs:/<run_id>/<model_folder> --no-conda -p 5002

export MLFLOW_TRACKING_URI=http://localhost:5001/

# from the client container
export MLFLOW_TRACKING_URI=http://172.17.0.2:5000

# problem with mismatched model versions between host and client container
# Try running models from the container
mlflow models serve --model-uri runs:/621ba09ea51048f68dc67485f0cb92c8/LinearRegression --no-conda -h 0.0.0.0  -p 5002


docker run -it --name client -p 5002:5002 mlflow_client

export MLFLOW_TRACKING_URI=http://172.17.0.2:5000

