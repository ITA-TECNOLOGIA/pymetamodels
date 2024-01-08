#!/usr/bin/python

def test_empty(y):
    # Test for empty variable as []
    
    if type(y) == type([]):
        if len(y) == 0:
            return False
        else:
            return True
    else:
        return True
    