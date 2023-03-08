# if MOTION_IMITATION_DIR variable is not set, then set it to the current working directory
if [ -z "$MOTION_IMITATION_DIR" ]; then
  MOTION_IMITATION_DIR=$(pwd)
fi

if [ -z "$MOTION_IMITATION_CONTAINER_NAME" ]; then
  MOTION_IMITATION_CONTAINER_NAME=MOTION_IMITATION_container
fi

echo "MOTION_IMITATION_DIR: $MOTION_IMITATION_DIR"
echo "MOTION_IMITATION_CONTAINER_NAME: $MOTION_IMITATION_CONTAINER_NAME"

# create ssh-key in MOTION_IMITATION_DIR without password
mkdir -p "$MOTION_IMITATION_DIR/.ssh"
ssh-keygen -t rsa -b 4096 -C "MOTION_IMITATION" -f "$MOTION_IMITATION_DIR/.ssh/id_rsa" -N ""

# CONNECT TO RUNNING DOCKER CONTAINER WITH THIS COMMAND
# ssh user@127.0.0.1 -p 2022 -i .ssh/id_rsa

# Set the environment variable for where the project is located
docker build --build-arg MOTION_IMITATION_DIR="$MOTION_IMITATION_DIR" -f "$MOTION_IMITATION_DIR"/docker/Dockerfile -t rlperf/motion_imitation:latest .

# If you do not have administrator rights to the host machine, remove the privileged command. Also do this if you only want to run headless mode.
xhost + local: # Allow docker to access the display
docker run --rm \
  -it \
  -p 2023:22 \
  -e DISPLAY="$DISPLAY" \
  -v "$HOME/.Xauthority:/user/.Xauthority:rw" \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  --name $MOTION_IMITATION_CONTAINER_NAME \
  --privileged \
  --gpus=all \
  rlperf/motion_imitation:latest
