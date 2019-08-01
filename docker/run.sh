# Run Docker with Display
docker run -it --runtime=nvidia --net=host --env="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" -v $(pwd):/mnt liaohaofu/adn
