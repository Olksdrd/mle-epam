# Containerization

### Model Description

This projects sets up a development environment inside a docker container to work on 
[IMDB movie reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) dataset using a pretrained version of [DistilBERT](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) transformer model.

### Repository Structure

### Notes on a Dockerfile

- The only environmental variable set in a Dockerfile is a timezone. The reason it's set so early is to allow `tzdata` package to be installed without asking questions, which just freezes docker build process. Default value can be overrided by -e TZ=time_zone flag in a docker run command.
- Python version inside the container is `3.10.12`. Changing the python-pip version to any other possible seems to require installing wget and tar packages, so I avoided it.
- dockerignore doesn't work with bind mounts, since all those files will be readded after the mount is established. So there is no way to get rid of some files (without restructuring the whole repo) either than deleting them once they are not needed anymore.

### Initialization

1. Copy `module2_docker` folder to a separate directory on your host
```
cp -r  module2_docker/ <project directory>
```
2. Navigate to that directory
```
cd <project directory>
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
5. Run `ls -l` command in the container shell. Now there are two options:
- Option 1. All the permissions were set up correcly and `/code` folder with all the files is owned by user.
- Option 2. All the files in `/code` are owned by the root. In this case run the following command in the container
```
chown -R $UID:$UID /code
```
Then go to your host and check under which UID the files are owned. On Ubuntu 22.04 with default settings it's most likely UID 100999. Then create a group with that GID on your host
```
sudo addgroup --gid 100999 <group name>
```
and add yourself to that group by running
```
sudo usermod -aG <group name> <your_username>
```
After that reboot/relogin for changes to take full effect. Now you should have access to all current files both from the container and from host filesystems.

There might be some problems with new files created after this, since that depends on the permission settings of the directory which may differ between machines. In that case either use sudo on your host to override user and group ownership or just run the following command, after cd to project directory either on host or in the container.
```
find . -type f -exec chmod 664 -- {} + \
  && find . -type d -exec chmod 775 -- {} +
```
Note that doing it on host will require sudo.

### Setting up git

Just run `git init` in the project root directory in the container. Due to bind mount all the changes should be syncronized immediately. To check that this is the case run `git config --get user.email` from container shell: it should show your user email that was configured on the host (at least in my case it did). If not, just add you username and email to git config.

Depending on some other settigs, it might also ask you to add the direcory path to safe.directory config. The command for that is provided by git in the error message. After that git should work as usual.

Also run `date` to check whether your timezone was set correctly. Otherwise your commit times might not match between the host and the container.

### Using the environment

#### Configuring VS Code

After exiting, container can be restarted with `docker start dev_container` and stopped with `docker start dev_container`.

Potentially more convenient way to use the environment is via CS Code "Dev Containers" package. After installing it, click on 'Open a remote window' and select 'Attach to a running container' option. Upon selecting dev_container (it should be running, obviously), new VS Code window will be opened. Open `/code` directory to get access to the project files in an IDE. You may need to install addtitonal extentions, like 'Python' to enable autocompletion. IDE setup needs to be done only once as long as you don't remove the container and just stop/restart it every time.

To close the environment just select 'Close remote connection' option and then stop the container.

#### Loading the Dataset

To load the dataset from Kaggle just run `load_dataset.py` script
```
python3 preprocessing/load_dataset.py
```
Csv file with raw data is stored in `data/raw_data` folder.

#### Loading the Model

To test that the transformer works on the data, run `batch_inference.py` script
```
python3 inference/batch_inference.py
```
This will load the model into container memory (only needs to be done once per container) and will also infer labels for first 20 reviews.

#### Settings and Environmental Variables

Time zone is the only environmental variable set in a Dockerfile. To reduce the chances of non-compatibility between different operating systems all the other were put in a `settings.json` file, which can be then loaded in a python script as a dictionary.

#### Logging

Logging is configuted via Python built-in logging module. All the logging configs are stored in a `logging/log_config.json`. By default logging is done both in the terminal and into `logging/logfile.log`. That can be changes simply by editing the config file.

#### Debugging

`debugpy` is preinstalled. In case some other modules are needed, jut pip install them.

