# Mlflow

### Module Description

Dataset description

### Repo Structure

```
.
├── basic_models
│   ├── GradientBoosting.py
│   ├── LinearRegression.py
│   └── RandomForest.py
├── data
│   └── data_loader.py
├── Dockerfile
├── Dockerfile_server
├── hpopt
│   ├── hbrv1.py
│   └── hbrv2.py
├── kaggle.json
├── MLproject
├── new_features
│   ├── GradientBoostingv2.py
│   └── RandomForestv2.py
├── python_env.yaml
├── README.md
├── requirements.txt
├── serving
│   ├── best_model.py
│   └── get_predictions.py
└── utils
    ├── configs.py
    ├── funcs.py
    └── __init__.py
```

### Running the Project

#### Step 1. Setting up Mlflow Server

1. Navigate to module 3 folder on your filesystem
```
cd <project_diretory>/module3_mlflow
```
2. Build and image for an mlflow server
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
6. Before running a client image it would be nice to set up mlflow tracking uri as an environmental variable. For that we need server container IP adress. We can get it either examiming the output of `docker inspect server` command or running (at least on Linux)
```
docker inspect server | grep -Po '(?<="IPAddress": ")[^"]*' | head -1
```
7. Run a client container
```
docker run -it -v ./:/code --name client -p 5002:5002 -e MLFLOW_TRACKING_URI=http://<server_IP>:5000/ mlflow_client
```
Alternatively, we can set up environmental variable in the *client* container to
```
export MLFLOW_TRACKING_URI=http://<server_IP>:5000/
```
Note that both containers are in one docker network, and from inside of that network server is at port 5000, whereas from the host it's at http://0.0.0.0:5001. Run something like `mlflow experiments search` in a client container shell to check that the connection is established properly.
```
docker run -it -v ./:/code --name client -p 5002:5002 -e MLFLOW_TRACKING_URI=http://172.17.0.2:5000/ mlflow_client
```
Note: sometimes this container for some reason eats half of my RAM. Then restarting and reattaching to it helps with this issue.

Check README for module 2 in case of any problems with file permissions.

#### Step 3. Running Mlflow Project

8. Import dataset from Kaggle by running `python data/data_loader.py`.

9. Lets run a simple linear regression just to have some baseline model.
```
mlflow run -e linear_regression --env-manager local --experiment-name Basic_Models --run-name lr .
```
The model is registered as Version 1. Observations from run results in mlflow ui:
- test R2 is around 0.64, which isn't that great
- there is saved residual plot in artifacts section. From it we can see some weird nonlinear pattern in residuals. I guess we could try some polynomial features (I did try and there was almost no improvement), but lets instead just run a more flexible nonlinear model.

10. Run histogram gradient boosting from sklearn (simply to avoid installing xgboost in a container) and see how it performs with default parameters
```
mlflow run -e boosting --env-manager local --experiment-name Basic_Models --run-name hbr .
```
- R2 is around 0.82, other metrics also improved dramatically compared to linear regression.
- nonlinear bend in a residuals is mostly gone. There's no obvious heteroscedasticity or other irregularities in residuals. This is much better.
- comparing test and training metrics, there is a clear sign of overfitting.

Note: I'm aware that there is no need in scaling for tree-based models (pipeline html representation is saved in artifacts). But removing it requires creating another pipeline, which is just not worth it.

11. Since an ensemble of trees performed much better, lets also try random forest (on sklearn defaults)
```
mlflow run -e forest --env-manager local --experiment-name Basic_Models --run-name rfr .
```
Observations:
- it took more time to run that histogram gradient boosting.
- test r2 and rmse are sligtly worse, but test mae is better that for boosting run.
- train r2 is around 0.975, and that's real strong overfitting.

Note that in all 3 models above I put tags, descriptions and aliases for model in the registry and not for individual runs. I think it's just easier to access them there, since they are all in a same place for different models and versions. I didn't log the dataset, since it's always the same, there are no data versions or stuff like that. And all the features version are stored via sklearn pipelines, which are serialized with cloudpickle (so all the custom classes from utils are also saved and we don't need to import them later).

12. Now lets try slightly regularizing histogram gradient boosted trees to deal with overfitting (I don't like regularizing random forest, since it takes more time to tune and from my experience almost always gets slighly behind boosting; so lets stick with one algorithm to tune just for simplicity)
```
mlflow run -e tuning_hbr --env-manager local --experiment-name HBR_Tuning --run-name hbrv1_tuning .
```
Hyperparameter tuning is done using hyperopt. Note that the number of trials can be changed in a `utils/configs.py` file. The default value is 30.
Observations:
- sorting all the run (they are in a separate experiment) by metrics shows that it didn't help at all
- the two parameters that were tuned were max_depth and l2_regularization. By selection all the runs in the ui, pressing compare and choosing contour plot for those two parameters on x and y axis and any test metric on z, we can see that the best values for l2 param is somewhere around 0. Small max_depth is also slightly better, but the difference in performance is negligble.

We could try regularizing with other parameters, but a more promising approach is to just create a couple new features.

13. I've added some feature interactions and bined several numerical features (unbinned versions aren't dropped, obviously). Html representation of a pipeline will be saved in run artifacts. Lets see how boosting performs now
```
mlflow run -e boostingv2 --env-manager local --experiment-name Bin_Inter_Feat --run-name hbrv2 .
```

14. And lets immediately run a random forest on a same features and then inspect how these two runs went
```
mlflow run -e forestv2 --env-manager local --experiment-name Bin_Inter_Feat --run-name rfrv2 .
```
Comparing all the metrics in the mlflow ui we can see that test R2 improved to by 0.02 to around 0.84 for a boosting and by a third of that for a random forest. Anyways, it seems like for datasets like this one it's better to invest more time into feature engineering than in hyperparameter tuning. But that's not the goal of this module, so instead lets try tuning histogram boosting one more time to get a slightly better final model. This time I slightly restricted range of (the same) hyperparameters closer to 0, based on a previous results (in retrospect, setting an upper range max_depth to 150 was a stupid idea: that sets essentially no restricions whereas the goal was to regularize). Obviosly, in general we need try more options for regularization, here I'm just checking how to integrate hyperparameter tuning with mlflow in principle.

15. Final hyperopt run. It's saved in the same experiment as a previous hyperopt runs
```
mlflow run -e tuning_hbrv2 --env-manager local --experiment-name HBR_Tuning --run-name hbrv2_tuning .
```
Note: all the runs are nested in one parent run called hbrv2_tuning. If they are not nested in your ui, that could be because you've previously sorted runs in this experiment and mlflow remembered that (just order by 'Created' columns).

#### Step 4. Serving the Best Model

16. Now lets serve the model with the best test R2. The following script finds the best hyperparameters from all runs and retrains the model with them `python serving/best_model.py`. Retraining is needed since I've decided not to register each hyperopt run. And it makes sense to have the best model registered.

17. Now the best model has alias 'prod', and we can simply serve it by running
```
mlflow models serve -m "models:/HistGradientBoostingRegressor@prod" --no-conda -h 0.0.0.0  -p 5002
```

18. Now we can send a request to localhost at port 5002 to get some predictions for a new data. Lets do it from a host shell. Run there `python serving/get_predictions.py` (need to navigate to module3_mlflow directory first obviously) to get predicted median house prices for 40 datapoints that we're (let pretend) interested in.

19. Now we can stop serving, stop client and server containers. Even if we remove them, all the data with runs will still persist in a named volume.


### Delete this:


Doesn't work:
```
mlflow models serve --model-uri runs:/f35101dc27794d75bf1bbac72cd1d202/HistGradientBoostingRegressor --no-conda -h 0.0.0.0  -p 5002
```
ModuleNotFoundError: No module named 'utils'

Works: (But need to set that alias and then also import parameters)
mlflow models serve -m "models:/HistGradientBoostingRegressor@prod" --no-conda -h 0.0.0.0  -p 5002
Then from host (don't forget to go to module3_mlflow folder)
`python serving/get_predictions.py`

Stop serving, close container, close server.

export MLFLOW_TRACKING_URI=http://localhost:5001/

mlflow run -e linear_regression --env-manager local --experiment-name Basic_Models --run-name lr .

mlflow run -e tuning --env-manager local --experiment-name tuning_cont --run-name hpopt .




mlflow models serve --model-uri runs:/
30a9daaed9124d33bc0aa8884e7eec67/30a9daaed9124d33bc0aa8884e7eec67-hbr --no-conda -p 5002


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

