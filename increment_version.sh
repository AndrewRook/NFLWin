#!/bin/bash

########################################################################
# This script intelligently increments NFLWin's version,
# based on the rules of semantic versioning.
# It does the following:
# 1. Parse command line arguments to determine whether to
#    increment major, minor, or patch version.
# 2. Makes sure it's not on the master branch.
# 3. Makes sure there aren't any changes that have been
#    staged but not committed.
# 4. Makes sure there aren't any changes that have been
#    committed but not pushed.
# 5. Makes sure all unit tests pass.
# 6. Compares current version in nflwin/_version.py to most recent
#    git tag to make sure they're the same.
# 7. Figures out what the new version should be.
# 8. Updates nflwin/_version.py to the new version.
# 9. Adds and commits nflwin/_version.py with commit message
#    "bumped [TYPE] version", where [TYPE] is major, minor, or patch.
#10. Tags latest commit with version number (no 'v').
#11. Pushes commits and tags.
########################################################################

set -e

#Parse command line arguments:
if [ "$#" -ne 1 ]; then
    echo "Syntax: ./increment_version.sh [major|minor|patch]"
    exit 1
fi

VERSION_TYPE=`echo "$1" | tr '[:upper:]' '[:lower:]'`

if [ "$VERSION_TYPE" != "major" -a "$VERSION_TYPE" != "minor" -a "$VERSION_TYPE" != "patch" ]; then
    echo "Version type must be one of 'major', 'minor', or 'patch'"
    exit 1
fi

#Ensure we're not on master:
CURRENT_BRANCH=`git rev-parse --abbrev-ref HEAD`
if [ "$CURRENT_BRANCH" == "master" ]; then
    echo "Must not be on master branch"
    exit 1
fi

#Make sure there aren't any staged changes:
STAGED_CHANGES_FLAG=`git status | grep "Changes to be committed" | wc -l`
if [ $STAGED_CHANGES_FLAG -ne 0 ]; then
    echo "Must not have any staged changes"
    exit 1
fi

#Make sure there aren't any unpushed changes:
git pull #Do this first to sync things

UP_TO_DATE_FLAG=`git status | sed -n 2p | grep "Your branch is up-to-date with" | wc -l`
if [ $UP_TO_DATE_FLAG -eq 0 ]; then
    echo "Must not have any unpushed changes"
    exit 1
fi

#Make sure all unit tests pass:
./run_tests.sh #Will return 1 if any tests fail, thus triggering the set -e flag.

#Get version in nflwin/_version.py
VERSION_PY = `grep "^__version__" nflwin/_version.py | awk '{print $NF}' | tr -d \"`

#Get version in github:


exit 0
