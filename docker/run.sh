# Start Docker with Display
docker run -it --runtime=nvidia --net=host --env="DISPLAY" -v="$HOME/.Xauthority:/root/.Xauthority:rw" -v $(pwd):/mnt --name myadn liaohaofu/adn
# Resume Docker with Display
docker exec -it -u root myadn /bin/bash
