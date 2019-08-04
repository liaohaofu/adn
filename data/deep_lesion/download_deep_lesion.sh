files="https://nihcc.box.com/shared/static/sp5y2k799v4x1x77f7w1aqp26uyfq7qz.zip
https://nihcc.box.com/shared/static/l9e1ys5e48qq8s409ua3uv6uwuko0y5c.zip
https://nihcc.box.com/shared/static/48jotosvbrw0rlke4u88tzadmabcp72r.zip
https://nihcc.box.com/shared/static/xa3rjr6nzej6yfgzj9z6hf97ljpq1wkm.zip
https://nihcc.box.com/shared/static/58ix4lxaadjxvjzq4am5ehpzhdvzl7os.zip
https://nihcc.box.com/shared/static/cfouy1al16n0linxqt504n3macomhdj8.zip
https://nihcc.box.com/shared/static/z84jjstqfrhhlr7jikwsvcdutl7jnk78.zip
https://nihcc.box.com/shared/static/6viu9bqirhjjz34xhd1nttcqurez8654.zip
https://nihcc.box.com/shared/static/9ii2xb6z7869khz9xxrwcx1393a05610.zip
https://nihcc.box.com/shared/static/2c7y53eees3a3vdls5preayjaf0mc3bn.zip"

files=($files)
num_files=${#files[@]}

for (( i=0; i<${num_files}; i++ ));
do
    idx=$(printf "%02d" "$(expr $i + 1)")
    if [ -f Images_png_$idx.zip ]; then
        echo "Images_png_$idx.zip exist"
    else
        wget ${files[$i]} -O Images_png_$idx.zip
        unzip Images_png_$idx.zip
    fi
done
echo "DeepLesion dataset downloaded!"