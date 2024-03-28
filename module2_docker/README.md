# Containerization

### Initialization

1. Copy module2_docker folder to a separate directory
```
cp -r  module2_docker/ ~/projects/hf
```
2. Navigate to that directory
```
cd ~/projects/hf
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
Then go to your host and check under which UID the files are owned. On default Ubuntu 22.04 it's most likely UID 100999. Then create a group with that GID on your host
```
sudo addgroup --gid 100999 <group name>
```
and add yourself to that group by running
```
sudo usermod -aG <group name> <your_username>
```
After that reboot/relogin for changes to take full effect. Now you should have access to all current files both from the container and from host filesystems.

There might be some problems with new files created after this, since that depends on the permission settings of the directory which may differ between machines. In that case either use sudo on your host to override user and group ownership or just run the folliwing command, after cd to project directory either on host or in the container.
```
find . -type f -exec chmod 664 -- {} + \
  && find . -type d -exec chmod 775 -- {} +
```
Note that doing it on host will require sudo.

### Setting up git

Just run `git init` in the project root directory either in the container or on host. Due to bind mount all the changes should be syncronized immediately, so it doesn't matter where to do it. To check that this is the case run `git config --get user.email` from container shell: it should show your user email that was configured on the host (at least in my case it did).

Depending on some other settigs, it might also ask you to add the direcory path to safe.directory config. The command for that is provided by git in the error message. After that git should work as usual.

Also run `date` to check whether your timezone was set correctly. Otherwise your commit times won't match between the host and the container.

### Using the environment

### Templates

Note that dockerignore doesn't work with bind mounts, so there is no way to get rid of some files either than deleting them.

