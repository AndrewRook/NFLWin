#!/bin/bash

set -e

sphinx-apidoc -f -o doc/source nflwin/ nflwin/tests
cd doc
make html
cd ../
