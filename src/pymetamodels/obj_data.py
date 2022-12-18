#!/usr/bin/env python3

import os,sys
from collections import OrderedDict

class objdata(dict):
    
    ## Common functions
    ###################
    
    def __init__(self,buff_type="OrderedDict",*arg,**kw):
        
        super(objdata, self).__init__(*arg, **kw)
        self.__dict__ = OrderedDict()
        self.buff_type = buff_type
      
    def __setitem__(self, key, item):
        self.__dict__[key] = item

    def __getitem__(self, key):
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __delitem__(self, key):
        del self.__dict__[key]

    def clear(self):
        return self.__dict__.clear()

    def copy(self):
        return self.__dict__.copy()

    def has_key(self, k):
        return k in self.__dict__

    def update(self, *args, **kwargs):
        return self.__dict__.update(*args, **kwargs)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def pop(self, *args):
        return self.__dict__.pop(*args)

    def __cmp__(self, dict_):
        return self.__cmp__(self.__dict__, dict_)

    def __contains__(self, item):
        return item in self.__dict__

    def __iter__(self):
        return iter(self.__dict__)

    def __unicode__(self):
        return unicode(repr(self.__dict__))
        
    def print(self, ident = '', braces=1, vdict = None):
        
        """ Recursively prints nested dictionaries."""
        
        if vdict is None:
            dictionary = self
        else:
            dictionary = vdict
            
        for key, value in dictionary.items():
            if isinstance(value, dict):
                print('  %s%s%s%s' %(ident,braces*'[',key,braces*']') )
                self.print(ident+'  ', braces+1, value)
            else:
                print("  "+ident+'%s = %s' %(key, value))        
        
    ## 
    ###################        