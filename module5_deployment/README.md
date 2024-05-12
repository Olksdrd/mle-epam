### Project Desctiption

### Repo Structure

```
.
├── dags
│   ├── batch_serving_dag.py
│   └── cali_housing_dag.py
├── dagtasks
│   ├── dagtasks
│   │   ├── cleaning.py
│   │   ├── __init__.py
│   │   ├── loading.py
│   │   ├── preprocessing.py
│   │   ├── serving.py
│   │   └── training.py
│   ├── README.md
│   └── setup.py
├── Dockerfile
├── .dockerignore
├── get_preds.py
├── README.md
├── requirements.txt
└── tests
    ├── batch_tests.py
    └── model_tests.py
```

### Model Training

1. Navigate to `module5_deployment` directory

2. Set up virtual environment (using conda or venv). For conda:

```
conda create --name <env_name> python pip
```
Then `conda activate <env_name>`. And install all the packages 
```
conda install -c conda-forge --file requirements.txt
```
3. Then install all functions needed to run an airflow workflow.
```
pip install -e ./dagtasks
```
4. Set up airflow home directory in a current folder `export AIRFLOW_HOME=$(pwd)` (or whatever is an equivalent in powershell).

5. Initialize a database by `airflow db init`.

6. (Optional) Check airflow.cfg file to verify that dags folder location in `dags_folder` variable is correct (the default path should be correct). Then set `load_examples = False` to make airflow UI less cluttered.

7. Next, start a scheduler by running `airflow scheduler`.

8. Open new terminal, navigate to module5_deployment and set up `export AIRFLOW_HOME=$(pwd)` in it again.

9. Create a new airflow user
```
airflow users create \
    --username user \
    --firstname user \
    --lastname user \
    --role Admin\
    --email user@email.com \
    --password 1234
```
10. Start airflow UI by running `airflow webserver -p 8080`

11. Open the UI in a browser at http://0.0.0.0:8080 and login.

12. Turn the dag on and run it in the UI.

13. First run `housing_hgb` DAG. But before that make sure that docker daemon is running, since the last task in a DAG builds a docker image for online serving.

### Batch Serving

1. Now run `batch_serving` DAG. Note that since we're using SequentialExecutor, running this DAG before the previous will lead to an infinite queue.

2. Once activated, the DAG will process a (pseudo)-new batch of data every 5 minutes.

### Online Serving

1. Run the following command to start serving the model
```
docker run -d --name server -p 5002:5002 mlflow 
```
2. It'll take around 5-10 second to start serving. After that we can send requests with new datapoints for prediction, as for example in a `get_preds.py` file. These results can be than redirected to a database or wherever they are needed.

### Additional Comments

- Airflow is not containerazed due to it running *extremely* bad and laggy inside containers on my laptop.
- In case if BashOperators don't work properly on your OS, they could be just skipped. The only thing they do are (a) building docker image (can be easily done manually) and (b) perform some simple unit test (nothing would happen in this toy example if they are skipped).
- Package doesn't include airflow DAGs themselves, since that would require changing airflow dag directory depending on the environment, which seems somewhat annoying.
- Unit tests are not packaged since it would just lead to unnecessary complications (maybe there is a simple way to make it work specifically with airflow, but I haven't found it).
- See module 4 for some additional notes on the pipeline.

