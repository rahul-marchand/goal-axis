"""Put this directory on PYTHONPATH to run any project script with the legend observation.

See goalmisgen/envs/legend.py. Imported by the interpreter at start-up, so it
reaches every subprocess a chain launches; it patches nothing until the maze
config module is actually imported.
"""

from goalmisgen.envs.legend import install_on_import

install_on_import()
