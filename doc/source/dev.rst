For Developers
=========================

This section of the documentation covers things that will be useful for those already contributing to NFLWin.

.. note::
   Unless stated otherwise assume that all filepaths given in this section start at the root directory for the repo. 

Testing Documentation
------------------------------------------

Documentation for NFLWin is hosted at `Read the Docs <https://readthedocs.org/>`_, and is built automatically when changes are made on the master branch or a release is cut. However, oftentimes it's valuable to display NFLWin's documentation locally as you're writing. To do this, run the following::

  $ ./build_local_documentation.sh

When that command finishes, open up ``doc/index.html`` in your browser of choice to see the site.

Updating the Default Model
--------------------------------------

NFLWin comes with a pre-trained model, but if the code generating that model is updated **the model itself is not**. So you have to update it yourself. The good news, however, is that there's a script for that::

  $ python make_default_model.py

.. note::
   This script hardcodes in the seasons to use for training and
   testing samples. After each season those will likely need to be
   updated to use the most up-to-date data.

.. note::
   This script requires ``matplotlib`` in order to run, as it produces a
   validation plot for the documentation.

Cutting a New Release
----------------------------------
NFLWin uses `semantic versioning <http://semver.org/>`_, which basically boils down to the following (taken directly from the webpage linked earlier in this sentence):

  Given a version number MAJOR.MINOR.PATCH, increment the:

  1. MAJOR version when you make incompatible API changes,
  2. MINOR version when you add functionality in a backwards-compatible manner, and
  3. PATCH version when you make backwards-compatible bug fixes.

Basically, unless you change something drastic you leave the major version alone (the exception being going to version 1.0.0, which indicates the first release where the interface is considered "stable").

The trick here is to note that information about a new release must live in a few places:

* In ``nflwin/_version.py`` as the value of the ``__version__`` variable.
* As a tagged commit.
* As a release on GitHub.
* As an upload to PyPI.
* (If necessary) as a documented release on Read the Docs.

Changing the version in one place but not in others can have relatively minor but fairly annoying consequences. To help manage the release cutting process there is a shell script that automates significant parts of this process::

  $ ./increment_version.sh [major|minor|patch]

This script does a bunch of things, namely:

1. Parse command line arguments to determine whether to
   increment major, minor, or patch version.
2. Makes sure it's not on the master branch.
3. Makes sure there aren't any changes that have been
   staged but not committed.
4. Makes sure there aren't any changes that have been
   committed but not pushed.
5. Makes sure all unit tests pass.
6. Compares current version in nflwin/_version.py to most recent
   git tag to make sure they're the same.
7. Figures out what the new version should be.
8. Updates nflwin/_version.py to the new version.
9. Uploads package to PyPI.
10. Adds and commits nflwin/_version.py with commit message
    "bumped [TYPE] version to [VERSION]", where [TYPE] is major, minor, or patch.
11. Tags latest commit with version number (no 'v').
12. Pushes commit and tag.

It will exit if **anything** returns with a non-zero exit status, and since it waits until the very end to upload anything to PyPI or GitHub if you do run into an error in most cases you can fix it and then just re-run the script. 

The process for cutting a release is as follows:

1. Make double sure that you're on a branch that's not ``master`` and you're ready to cut a new release (general good practice is to branch off from master *just* for the purpose of making a new release).
2. Run the ``increment_version.sh`` script.
3. Fix any errors, then rerun the script until it passes.
4. Make a PR on GitHub into master, and merge it in (self-merge is ok if branch is just updating version).
5. Make release notes for new release on GitHub.
6. (If necessary) go to Read the Docs and activate the new release.
