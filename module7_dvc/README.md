### Rinning the project

1. Navigate to `module7_dvc` directory.

2. Build a docker image
```
docker build -t dvc .
```
3. Start a container
```
docker run -it --name dvc_cont --network=host --rm dvc
```
5. Set up git inside a container
```
git config --global user.email "you@example.com" && \
git config --global user.name "Your Name" && \
git init && \
git add . && \
git commit -m 'Initial commit"
```
5. Download a version of the dataset that corresponds to md5 hash in a `data/housing.csv.dvc` file
```
dvc pull data/housing.csv
```
It'll ask you to login into gmail account from your browser. (Host network is needed to pass the redirect back to the container.)

6. Inspect the first pipeline stages
```
dvc dag prediction_pipe/dvc.yaml
```
7. Navigate to `prediction_pipe/` directory to avoid constantly specifying a path to `dvc.yaml` file.
8. Lets run some experiments on the pipeline
```
dvc exp run -S 'training.n_bins=20'
```
8. Commit the results to git
```
git add dvc.lock params.yaml 
```
And do a git commit.

9. Lets try some other hyperparameter values
```
dvc exp run -S 'training.n_bins=5' -S 'training.l2_regularization=0.01'
```
10. Now we can compare this run to the previous commited run with
```
dvc params diff
```
```
dvc metrics diff
```
(There is a VS code extension that does all these way easily)

11. Navigate to `anomaly_detection_pipe/`.
12. Inspect the pipeline stages
```
dvc dag dvc.yaml
```
This pipeline reuses 2 initial stages from the previous pipe.
13. Run the second pipeline by `dvc repro`.
14. You can use `dvc push` to save all the runs to dvc repo so there is no need to redo then again on host or in a different container.

