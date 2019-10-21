
#!/bin/bash

mflag=" -m"
mflagval=" not requires_db"
python_version=`python -c "import sys; print(sys.version_info.major)"`
pytest_command="py.test"
if [ $python_version == "3" ]; then
    pytest_command="pytest"
fi

while getopts ":d" opt; do
    case $opt in
	d)
	    echo "Running all tests..."
	    mflagval=''
	    mflag=''
	    ;;
	\?)
	    echo ""
	    echo ""
	    echo "Invalid option: -$OPTARG"
	    echo "Usage:"
	    echo "-----------------"
	    echo "-d: run tests which require nfldb database access"
	    echo ""
	    ;;
    esac
done

python -m ${pytest_command}${mflag}"${mflagval}" --cov=nflwin --cov-config .coveragerc --cov-report term-missing nflwin/tests/
