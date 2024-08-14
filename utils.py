# imports
import geojson
import json
import yaml

def format_time(seconds):
    return f'{int(seconds // 60)} minute(s) {int(seconds % 60)} second(s)'

def read_geojson(path):
    with open(path) as geojson_file:
        return geojson.load(geojson_file)

def read_json(path):
    with open(path, 'r') as json_file:
        return json.load(json_file)

def read_yaml(path):
    with open(path, 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

def str_to_bool(string):
    '''convert a string input to a Boolean variable'''
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    return False
