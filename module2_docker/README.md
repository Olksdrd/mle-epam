# Containerization

### Model Description

This projects sets up a development environment inside a docker container to work on 
[IMDB movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) dataset using a pretrained version of [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) transformer model.

### Repository Structure

```
module2_docker/
├── Dockerfile
├── inference
│   └── batch_inference.py
├── kaggle.json
├── logging
│   └── log_config.json
├── preprocessing
│   └── load_dataset.py
├── README.md
├── requirements.txt
└── settings.json
```

### Notes on a Dockerfile

- The only environmental variable set in a Dockerfile is a timezone. The reason it's set so early is to allow `tzdata` package to be installed without asking questions, which just freezes docker build process. Default value can be overrided by -e TZ=time_zone flag in a docker run command.
- Python version inside the container is `3.10.12`. Changing the python-pip version to any other possible seems to require installing wget and tar packages, so I avoided it ot keep image less complex.
- dockerignore [doesn't work](https://github.com/docker/compose/issues/2098) with bind mounts, since all those files will be readded after the mount is established. So there is no way to get rid of some files (without restructuring the whole repo) either than deleting them once they are not needed anymore.

### Initialization

There has to be a version control system for the dev environment. But this module is already a part of git repo and having nested .git folders is not the best idea. So, there are several options. Either to set up git submodules or just initiate dev environment in a completely new folder. The first option seems like an unnecessary complexity, since project of developing and using the dev environment are unrelated and I don't think they should be managed through the same .git folder.

Steps to initialize dev environment:

1. Copy `module2_docker` folder to a separate directory on your host
```
cp -r  module2_docker/ <project_directory>
```
2. Navigate to that directory
```
cd <project_directory>
```
3. Build the docker image
```
docker build -t devenv:1.0 \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) .
```
4. Run a docker container
```
docker run --name dev_container -it -v $(pwd):/code devenv:1.0
```
#### Permissions on Linux

5. Run `ls -l` command in the container shell. Now there are two options:
- Option 1. All the permissions were set up correcly and `/code` folder with all the files is owned by a user named user (by default; use --build-arg to change the name).
- Option 2. All the files in `/code` are owned by the root.

For the [option 2](https://github.com/docker/desktop-linux/issues/31) case, run the following command in the container
```
chown -R $UID:$UID /code
```
Now go to your host and check under which UID the files are owned. On Ubuntu 22.04 with default settings it's most likely UID 100999. Then create a group with that GID on your host
```
sudo addgroup --gid 100999 <group_name>
```
and add yourself to that group by running
```
sudo usermod -aG <group_name> <your_username>
```
After that reboot/relogin for changes to take full effect. Now you should have access to all current files both from the container and from host filesystems.

There might be some problems with new files created after this, since that depends on the default permission settings of the directory which may differ between machines. In that case either use sudo on your host to override user and group ownership for those specific files/folders or just run the following command, after cd to project directory either on host or in the container.
```
find . -type f -exec chmod 664 -- {} + \
  && find . -type d -exec chmod 775 -- {} +
```
Note that doing it on host may require sudo.

### Setting up git

Just run `git init` in the project root directory in the container. Due to bind mount all the changes should be synchronized immediately. After that you may need to configure username and email.

Depending on some other settings, it might also ask you to add the direcory path to safe.directory config. The command for that is provided by git in the error message. After that git should work as usual.

Also run `date` to check whether the timezone was set correctly. Otherwise commit times might not match between the host and the container.

### Using the Environment

#### Configuring VS Code

After exiting, container can be restarted with `docker start dev_container` and stopped with `docker stop dev_container`.

Potentially more convenient way to use the environment is via CS Code 'Dev Containers' package. After installing it, click on 'Open a remote window' in the bottom left corner and select 'Attach to a running container' option. Upon choosing dev_container (it should be running, obviously), a new VS Code window will be opened. Navigate to `/code` directory to get access to the project files in an IDE. You may need to install addtitonal extentions, like 'Python' to enable autocompletion. IDE setup needs to be done only once as long as you don't remove the container and just stop/restart it every time.

To close the environment just select 'Close remote connection' option and then stop the container.

#### Loading the Dataset

To load the dataset from Kaggle just run `load_dataset.py` script
```
python3 preprocessing/load_dataset.py
```
Csv file with raw data is stored in `data/raw_data` folder.

Data loader script uses my Kaggle API key stored in `kaggle.json` to avoid manually entering one, so lets hope nobody finds it here. And I'll reset it later.

#### Loading the Model

To test that the transformer works on the data, run `batch_inference.py` script
```
python3 inference/batch_inference.py
```
This will load the model into container memory (only needs to be done once per container) and will also infer labels for first 20 reviews. Note that trying to predict $\gg 100$ datapoints at once takes some time and lots of RAM.

#### Settings and Environmental Variables

To reduce the chances of non-compatibility between different operating systems all the global project-wise settings were put in a `settings.json` file, which can be then loaded in a python script as a dictionary. Time zone is the only environmental variable set in a Dockerfile.

#### Logging

Logging is configuted via Python built-in logging module. All the logging configs are stored in a `logging/log_config.json`. By default logging is done both in a terminal and saved into a `logging/logfile.log` file. This can be changes simply by editing the config file.

#### Debugging

`debugpy` is preinstalled. In case some other modules are needed, jut pip install them.

