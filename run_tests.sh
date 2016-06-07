
#!/bin/bash

mflag=" -m"
mflagval=" not requires_db"

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

python -m py.test${mflag}"${mflagval}" --cov=nflwin --cov-config .coveragerc --cov-report term-missing nflwin/tests/
