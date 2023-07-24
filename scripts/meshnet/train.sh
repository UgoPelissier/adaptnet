clear

for arg in "$@"
do
   key=$(echo $arg | cut -f1 -d=)

   key_length=${#key}
   value="${arg:$key_length+1}"

   export "$key"="$value"
done

cd $PYTHONPATH
python meshnet/main.py fit -c meshnet/configs/${env}.yaml