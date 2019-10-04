import tables
import os

def open_data_file(filename, readwrite="r"):
    return tables.open_file(filename, readwrite)
