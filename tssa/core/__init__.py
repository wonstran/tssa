'''
This file is used to add the parent directory to the sys.path so that the modules in the parent directory can be imported.
'''
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))