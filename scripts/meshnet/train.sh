clear

if [ "$1" == "-h" ]; then
    echo "Usage: ./stokes2.sh -c <c>"
    echo "c: Configuration file"
    exit 0
fi

while getopts c: flag
do
    case "${flag}" in
        c) c=${OPTARG};;
    esac
done

cd $PYTHONPATH
python meshnet/main.py fit -c meshnet/configs/${c}.yaml