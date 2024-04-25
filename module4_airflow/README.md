1. Navigate to module4 directory

2. Set up virtual environment (using conda or venv)
```
conda create --name <env_name> python pip
```
conda activate <env_name>

conda install -c conda-forge --file requirements.txt

2. export AIRFLOW_HOME=$(pwd)

3. airflow db init

4. airflow scheduler

5. Check dags folder

6. Set `load_examples = False`

7. Open new terminal and navigate to module4

8. export AIRFLOW_HOME=$(pwd)

9. 
```
airflow users create \
    --username admin \
    --firstname Oleksandr \
    --lastname Drozdov \
    --role Admin --email alexua.drozdow@gmail.com
```
And then enter password
10. airflow webserver -p 8080

11. Open the UI in a browser at and login.

11. Turn the dag on and run it in th UI

Put python file with dags into dags folder. See path on top of airflow.cfg file

airflow dags list-import-errors


5. Sensor
6. Branch
9. Dockstrings
10. Naming
11. Tags
12. Readme