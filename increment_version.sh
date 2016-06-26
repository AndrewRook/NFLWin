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
# 9. Uploads package to PyPI.
#10. Adds and commits nflwin/_version.py with commit message
#    "bumped [TYPE] version to [VERSION]", where [TYPE] is major, minor, or patch.
#11. Tags latest commit with version number (no 'v').
#12. Pushes commit and tag.
########################################################################

set -e

echo "Need to change pypi server!
exit 1

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
VERSION_PY=`grep "^__version__" nflwin/_version.py | awk '{print $NF}' | tr -d \"`

#Get version in git:
VERSION_GIT=`git describe --tags $(git rev-list --tags --max-count=1)`

#Ensure versions are the same:
if [ $VERSION_PY != $VERSION_GIT ]; then
    echo "Versions must match! Python version=${VERSION_PY}, git version=${VERSION_GIT}"
    exit 1
fi

#Determines what new version should be:
MAJOR=`echo $VERSION_PY | awk -F"." '{print $1}'`
MINOR=`echo $VERSION_PY | awk -F"." '{print $2}'`
PATCH=`echo $VERSION_PY | awk -F"." '{print $3}'`
if [ $VERSION_TYPE == "patch" ]; then
    PATCH=$(expr $PATCH + 1)
elif [ $VERSION_TYPE == "minor" ]; then
    MINOR=$(expr $MINOR + 1)
    PATCH=0
else
    MAJOR=$(expr $MAJOR + 1)
    MINOR=0
    PATCH=0
fi
NEW_VERSION="$MAJOR.$MINOR.$PATCH"

#Update nflwin/_version.py:
sed -i.bak "s/${VERSION_PY}/${NEW_VERSION}/" nflwin/_version.py
rm nflwin/_version.py.bak

#Upload package to PyPI:
python setup.py sdist upload -r pypitest

#Stage and commit nflwin/_version.py
git add nflwin/_version.py
git commit -m "bumped ${VERSION_TYPE} version to ${NEW_VERSION}"

#Tag the commit:
git tag -a ${NEW_VERSION} -m "bumped ${VERSION_TYPE}"

#Push the commit and tag:
git push
git push origin ${NEW_VERSION}

echo "finished!"

exit 0
