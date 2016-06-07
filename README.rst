NFLWin
-------------

A package designed to compute Win Percentage (WP) for NFL plays.

Installation Instructions
---------------------------------------------

1. Installing nfldb (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
These steps are only necessary if you want to re-build the model or
use nfldb as your data source. Most of these instructions can be found on the nfldb page, but
I've repeated them here in the order I followed them (on Mac OS 10.10).

Installing Postgres
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I had an old install kicking around, so I first had to clean that up.
Since I was using Homebrew::

  $ brew uninstall -force postgresql
  $ rm -rf /usr/local/var/postgres/ # where I'd installed the prior DB

Then install a fresh version::
  
  $ brew update
  $ brew install postgresql


Start Postgres and Create a Default DB
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
You can choose to run Postgres at startup, but I don't use it that
often so I choose not to do those steps - I just run it in the
foreground with this command::

  $ postgres -D /usr/local/var/postgres

Or in the background with this command::
  
  $ pg_ctl -D /usr/local/var/postgres -l logfile start

If you don't create a default database based on your username,
launching Postgres will fail with a ``psql: FATAL:  database
"USERNAME" does not exist`` error::

  $ createdb `whoami`

Check that the install and configuration went well by launching
Postgres as your default user::

  $ psql
  psql (9.5.2)
  Type "help" for help.

  USERNAME=#

Next, add a password::

  USERNAME=# ALTER ROLE "USERNAME" WITH ENCRYPTED PASSWORD 'choose a
  superuser password';
  USERNAME=# \q;

Edit the ``pg_hba.conf``file found in your database (in this case
``/usr/local/var/postgres/pg_hba.conf``), and change all instances of
``trust`` to ``md5``. 

Create nfldb Postgres User and Database
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Start by making a user::

  $ createuser -U USERNAME -E -P nfldb

where you replace ``USERNAME`` with your actual username. Make up a
new password. Then make the nfldb database::

  $ createdb -U USERNAME -O nfldb nfldb

You'll need to enter the password for the USERNAME account. Next, add
the fuzzy string matching extension::

  $ psql -U USERNAME -c 'CREATE EXTENSION fuzzystrmatch;' nfldb

You should now be able to connect the nfldb user to the nfldb
database::

  $ psql -U nfldb nfldb

From this point you should be able to follow along with the
instructions from `nfldb
<https://github.com/BurntSushi/nfldb/wiki/Installation#importing-the-nfldb-database>`_. 

2. Setting up Your Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you're going to use ``nfldb`` to get your game data, it's critical
to complete part 1 **before** moving to this part, or else you may
find weird things are broken (for instance, having to
uninstall/reinstall ``psycopg2``). 

2a. Developers
^^^^^^^^^^^^^^^^^^^^^^^^^^
Pretty simple - get ``conda``, then::

  $ conda env create -f environment.yml

2b. Users
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This will depend on whether or not you want the nfldb dependencies.


3. Running Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``NFLWin`` uses ``pytest`` to run unit tests. You can invoke ``pytest``
like so::

  $ ./run_tests.sh

This will run the test suite on all modules and print out the results
as well as a coverage report. If you have ``nfldb`` properly installed
and wish to run additional tests specifically relating to querying the
database (will take longer), use the ``-d`` flag::

  $ ./run_tests.sh -d

