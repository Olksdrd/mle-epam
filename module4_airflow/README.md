### Repo Structure

```
.
├── dags
│   ├── cali_housing_dag.py
│   └── utils
│       ├── cleaning.py
│       ├── __init__.py
│       ├── loading.py
│       ├── preprocessing.py
│       └── training.py
├── README.md
└── requirements.txt
```

### Running the project

1. Navigate to `module4_airflow` directory

2. Set up virtual environment (using conda or venv). For conda:

```
conda create --name <env_name> python pip
```
Then `conda activate <env_name>`. And install all the packages 
```
conda install -c conda-forge --file requirements.txt
```

3. Set up airflow home directory in a current folder `export AIRFLOW_HOME=$(pwd)` (or whatever is an equivalent in powershell).

4. Initialize a database by `airflow db init`.

5. (Optional) Check airflow.cfg file to verify that dags folder location in `dags_folder` variable is correct (the default path should be correct). Then set `load_examples = False` to make airflow UI less cluttered.

6. Next, start a scheduler by running `airflow scheduler`.

7. Open new terminal, navigate to module4_airflow and set up `export AIRFLOW_HOME=$(pwd)` in it again.

8. Create a new airflow user
```
airflow users create \
    --username admin \
    --firstname <name> \
    --lastname <name> \
    --role Admin \
    --email <email> \
    --password 1234
```
```
airflow users create \
    --username admin \
    --firstname Oleksandr \
    --lastname Drozdov \
    --role Admin \
    --email alexua.drozdow@gmail.com \
    --password 1234
```
9. Start airflow UI by running `airflow webserver -p 8080`

10. Open the UI in a browser at http://0.0.0.0:8080 and login.

11. Turn the dag on and run it in the UI.

### Comments on the pipeline

- Since the dataset used in a previous module is too clean, I've manually made it less clean just so there are things to fix further in a pipeline. Obviously, this is redundant, but that allowed me to just stick with the dataset I used previously.
- All intermediate transformations are saved into csv files in a data/ folder. Splitting it into several tasks and saving/loading makes the pipeline somewhat slower, but also more modular (with each task doing more or less one thing or at least a set of related operations) so it's (hopefully) easier to rework it by simply removing some tasks if they are not needed. 
- Feature interactions are added before train/test split since they are all row-wise operations that don't involve the target variable, so there shouldn't be any data leakage here.
- Since there can be data leakage between cross-validation folds in a training pipeline (say, due to the use of K-means for binning features), we need to perform cross-validation on an entire pipeline. So it's just way more convenient to leave the sklearn pipeline as a one piece and not to split into several tasks. This allows to avoid saving and loading all the intermediate steps.
