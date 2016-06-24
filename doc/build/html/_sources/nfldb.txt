Using Data From nfldb
-------------

NFLWin comes with robust support for querying data from `nfldb
<https://github.com/BurntSushi/nfldb>`_, a package designed to
facilitate downloading and accessing play-by-play data. There are
functions to query the nfldb database in :py:mod:`nflwin.utilities`,
and :py:class:`nflwin.model.WPModel` has keyword arguments that allow
you to directly use nfldb data to fit and validate a WP model. Using
nfldb is totally optional: a default model is already fit and ready to
use, and NFLWin is fully compatible with any source for play-by-play
data. However, nfldb is one of the few free sources of up-to-date NFL
data and so it may be a useful resource to have. 

nfldb is pip-installable, and can be installed as an extra dependency
(``pip install nflwin[nfldb]``). Without setting up the nfldb
Postgres database first, however, the pip install will succeed but
nfldb will be unuseable. What's more, trying to set up the database
*after* installing nfldb may fail as well. 

The nfldb wiki has `fairly decent installation instructions
<https://github.com/BurntSushi/nfldb/wiki/Installation>`_, but I know
that when I went through the installation process I had to interpret
and adjust several steps. I'd at least recommend reading through the
wiki first, but in case it's useful 
I've listed the steps I followed below (for reference I was on Mac OS 10.10).

Installing Postgres
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
I had an old install kicking around, so I first had to clean that up.
Since I was using `Homebrew <http://brew.sh/>`_::

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

Edit the ``pg_hba.conf``file found in your database (in my case the
file was 
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


