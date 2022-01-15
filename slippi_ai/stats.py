import collections
import pandas
import sqlite3
import melee
from slippi_ai import paths
import os
cwd = os.getcwd()
paths.DB_PATH = cwd+'/data/melee_public_slp_dataset.sqlite3'
conn = sqlite3.connect(paths.DB_PATH)
TABLE = pandas.read_sql_query("SELECT * from replays", conn)
char_col = 'in_game_character' if 'in_game_character_0' in TABLE.columns else 'css_character'
# TODO - Track multiple versions of sql database, make char_col more general and throw exception if missing

def get_all_names():
  return TABLE.filename

def get_fox_ditto_names():
  table = TABLE
  table = table[table.css_character_0 == melee.Character.FOX.value]
  table = table[table.css_character_1 == melee.Character.FOX.value]
  return table.filename

BAD_NAMES = set([
    # filtered by Gurvan
    '20200212 - HNC 15 - PM 1035 - Fox (Green) vs Marth (Green) - Fountain of Dreams.slp',
    'Game_20190505T115119.slp',
    'Game_20190505T120725.slp',
    'Mewtwo vs Bowser [] Game_00224CCD4B5E_20200311T193709.slp',
    'Mewtwo vs Bowser [] Game_00224CCD4B5E_20200311T193815.slp',
    # found to have changing player port mid-game!
    'Game_20190505T184140.slp',
    'Game_20190505T095856.slp',
    '20_38_38 Fox + Falco (FD).slp',
    'Game_20190505T151309.slp',
    'Game_20190505T182248.slp',
])

SUBSETS = {
  "fox_dittos": get_fox_ditto_names,
  "all": get_all_names,
}

def get_subset(subset):
  names = SUBSETS[subset]()
  return [name for name in names if name not in BAD_NAMES]

def to_name(c):
  return melee.Character(c).name

def print_matchups():
  p0 = map(to_name, table[f'{char_col}_0'])
  p1 = map(to_name, table[f'{char_col}_1'])

  matchups = map(tuple, map(sorted, zip(p0, p1)))
  matchups = collections.Counter(matchups)

  sorted_matchups = sorted(matchups.items(), key=lambda kv: kv[1])
  total_games = len(table)

  for (c1, c2), v in sorted_matchups:
    print(c1.rjust(13), c2.rjust(13), f'{100*v/total_games:.2f}')

if __name__ == '__main__':
  print_matchups()
