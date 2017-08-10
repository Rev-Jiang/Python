#-*- coding: UTF-8 -*-
#上句表示可用中文注释，否则默认ASCII码保存

# Filename : dataanalysis.py
# author by : Rev_997

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def isiterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:#not iterable
        return False

#if it is not list or NumPy, transfer it
if not isinstance(x,list) and isiterable(x):
    x=list(x)

#is and is not are used to judge if the varible is None, as None is unique.
a=None
a is None

import datetime
dt=datetime(2011,10,29,20,30,21)
dt.day
dt.minute

dt.date()
dt.time()
#datetime could be transfered to string by function striftime
dt.strftime('%m/%d/%Y %H:%M')
#string could be transfered to datetime by function strptime
datetime.strptime('20091031','%Y%m%d')

#substitute 0 for minutes and seconds 
dt.replace(minute=0,second=0)

#the difference of two datetime objects produce a datetime.timedelta
dt2=datetime(2011,11,15,22,30)
delta=dt2-dt
delta
type(delta)
#add a timedelta to a datetime -- get a now datetime
dt+delta

#if elif else
if x:
    pass
elif:
    pass
else:
    pass

#for
for value in collection:
    #do something wuth value
    #continue
    #break

for a,b,c in iterator:
    #do something
    
#while
x=256
total=0
while x>0:
    if total>500:
        break
    total+=x
    x=x//2



def attempt_float(x):
    try:
        return float(x)
    except:
        return x
#once the float(x) is invalid, the except works


def attempt_float(x):
    try:
        return float(x)
    except(TypeError,ValueError):
        return x
#catch the abnormity


#value=true-expr if condition else false-expr
#same as
'''
if condition:
    value=true-expr
else:
    value=false-expr
'''

#about tuple
tup=4,5,6
tup
#(4,5,6)
#transfer to tuple
tuple([4,0,2])
tuple('string')

#tuple use + to generate longer tuple

#tuple.append()
#tuple.count()

#list.append()
#list.insert()
#list.pop()
#list.remove()
#list.extend()
#list.sort()

import bisect
c=[1,2,2,2,3,4,7]
#find the suitable position 
bisect.bisect(c,2)
#insert the new number
bisect.insort(c,6)
###attention: bisect is suitable for ordered sequence

#----------------------------------------------------------------
#some function of list

#enumerate 
for i,value in enumerate(collection):
    #do something with value

some_list=['foo','bar','baz']
mapping=dict((v,i) for i,v in enumerate(some_list))
mapping

#sorted
sorted([7,2,4,6,3,5,2])
sorted('horse race')
#powerful with set
sorted(set('this is just some string'))

#zip
seq1=['foo','bar','baz']
seq2=['one','two','three']
zip(seq1,seq2)
seq3=[False,True]
zip(seq1,seq2,seq3)
#several arrays iterate together with zip
for i,(a,b) in enumerate(zip(seq1,seq2)):
    print('%d: %s, %s' % (i,a,b))
#unzip
pitchers=[('Nolan','Ryan'),('Roger','Clemens'),('Schilling','Curt')]
first_names,last_names=zip(*pitchers)# * is meant zip(seq[0],seq[1],...,seq[len(seq)-1]) 
first_names
last_names

#reversed
list(reversed(range(10)))

#dictionary
empty_dict={}d1={'a':'some value','b':[1,2,3,4]}
d1
#delete 
del d1[5]
#or
ret=d1.pop('dummy')
ret
#get keys and values
d1.keys()
d1.values()
#combine two dictionaries
d1.update({'b':'foo','c':12})
d1
#match two list to be dictionary
'''
mapping={}
for key,value in zip(key_list,value_list):
    mapping[key]=value
'''
mapping=dict(zip(range(5),reversed(range(5))))
mapping
#brief way to express circulation by dict
'''
if key in some_dict:
    value=some_dict[key]
else:
    value=default_value
'''
value=some_dict.get(key,default_values)
#the vlaue of dictionary is set as other list
'''
words=['apple','bat','bar','atom','book']
by_letter={}
for word in words:
    letter=word[0]
    if letter not in by_letter:
        by_letter[letter]=[word]
    else:
        by_letter[letter].append(word)
by_letter
'''
by_letter.setdefault(letter,[]).append(word)
#or use defaultdict class in Module collections
from collections import defaultdict
by_letter=defaultdict(list)
for word in words:
    by_letter[word[0]].append(word)
#the key of dictionary should be of hashability--unchangable
hash('string')
hash((1,2,(2,3)))
hash((1,2,[3,4]))#no hashability as list is changable
#to change a list to tuple is the easiest way to make it a key
d={}
d[tuple([1,2,3])]=5
d

#set
set([2,2,2,1,3,3])
{2,2,2,1,3,3}
a={1,2,3,4,5}
b={3,4,5,6,7,8}
#intersection
a|b
#union
a&b
#difference
a-b
#symmetric difference
a^b
#if is subset
a_set={1,2,3,4,5}
{1,2,3}.issubset(a_set)
a_set.issuperset({1,2,3})
#set could use the == to judge if the same
{1,2,3}=={3,2,1}
#the operation of the sets
a.add(x)
a.remove(x)
a.union(b)
a.intersection(b)
a.difference(b)
a.symmetric_difference(b)
a.issubset(b)
a.issuperset(b)
a.isdisjoint(b)

#the derivative of list&set&dictionary
'''
[expr for val in collection if condition]
is the same as
result=[]
for val in collection:
    if condition:
        result.append(expr)
'''
#list
#[expr for val in collection if condition]
strings=['a','as','bat','car','dove','python']
[x.upper() for x in strings if len(x)>2]
#dicrionary
#dict_comp={key-expr:value-expr for value in collection if condition}
loc_mapping={val:index for index, val in enumerate(string)}
loc_mapping
#or
loc_mapping=dict((val,idx) for idx, val in enumerate(string))
#set
#set_comp={expr for value in collection if condition}
unique_lengths={len(x) for x in strings}
unique_lengths

#list nesting derivative
all_data=[['Tom','Billy','Jeffery','Andrew','Wesley','Steven','Joe'],
          ['Susie','Casey','Jill','Ana','Eva','Jennifer','Stephanie']]
#find the names with two 'e' and put them in a new list
names_of_interest=[]
for name in all_data:
    enough_es=[name for name in names if name.count('e')>2]
    names_of_interest.extend(enough_es)
#which could be shorten as below:
result=[name for names in all_data for name in names
        if name.count('e')>=2]
result
#flat a list consist of tuples
some_tuples=[(1,2,3),(4,5,6),(7,8,9)]
flattened=[x for tup in some_tuples for x in tup]
flattened
'''
flattened=[]
for tup in some_tuples:
    for x in tup:
        flattened.append(x)
'''
#which is different from:
[[x for x in tup] for tup in some_tuples]

#clean function
import re
def clean_strings(strings):
    result=[]
    for value in strings:
        value=value.strip()
        value=re.sub('[!#?]','',value) #Remove punctuation marks
        value=value.title()
        result.append(value)
    return result
states=[' Alabama ','Georgia!','Georgia','georgia','FlOrIda','south   carolina##','West virginia?']
clean_strings(states)
#or
def remove_punctuation(value):
    return re.sub('[!#?]','',value)
clean_ops=[str.strip,remove_punctuation,str.title]
def clean_strings(strings,ops):
    result=[]
    for value in strings:
        for function in ops:
            value=function(value)
        result.append(value)
    return result
clean_strings(states,clean_ops)

#anonymous function
#lambda [arg1[, arg2, ... argN]]: expression
#exmaple 1
#use def define function
def add( x, y ):
  return x + y
#use lambda expression
lambda x, y: x + y
#lambda permits default parameter
lambda x, y = 2: x + y
lambda *z: z
#call lambda function
a = lambda x, y: x + y
a( 1, 3 )
b = lambda x, y = 2: x + y
b( 1 )
b( 1, 3 )
c = lambda *z: z
c( 10, 'test')
#example2
#use def define function
def add( x, y ):
  return x + y
#use lambda expression
lambda x, y: x + y
#lambda permits default parameter
lambda x, y = 2: x + y
lambda *z: z 
#call lambda function
a = lambda x, y: x + y
a( 1, 3 )
b = lambda x, y = 2: x + y
b( 1 )
b( 1, 3 )
c = lambda *z: z
c( 10, 'test')
#example 3
def apply_to_list(some_list,f):
    return [f(x) for x in some_list]
ints=[4,0,1,5,6]
apply_to_list(ints,lambda x:x*2)
#example 4
strings=['foo','card','bar','aaaa','abab']
strings.sort(key=lambda x: len(set(list(x))))
strings

#currying
'''
def add_numbers(x,y):
    return x+y
add_five=lambda y:add_numbers(5,y)
'''
#partial function is to simplify the process
from functools import partial
add_five=partial(add_numbers,5)

#generator expression
gen=(x**2 for x in xxrange(100))
gen
#the same:
def _make_gen():
    for x in xrange(100):
        yield x**2
gen=_make_gen()
#generator expression could be used in any python function acceptable of generator
sum(x**2 for x in xrange(100))
dict((i,i**2) for i in xrange(5))

#itertools module
import itertools
first_letter=lambda x:x[0]
names=['Alan','Adam','Wes','Will','Albert','Steven']
for letter,names in itertools.groupby(names,first_letter):
    print letter,list(names) #names is a genetator
#some functions in itertools
imap(func,*iterables) 
ifilter(func,iterable)
combinations(iterable,k)
permutations(iterable,k)
groupby(iterable[,keyfunc])

#documents and operation system
path='xxx.txt'
f=open(path)
for line in f:
    pass
#remove EOL of every line
lines=[x.rstrip() for x in open(path)]
lines
#set a empty-lineproof doc
with open('tmp.txt','w') as handle:
    handle.writelines(x for x in open(path) if len(x)>1)
open('tmp.txt').readlines()
#some function to construct documents
read([size])
readlines([size])
write(str)
close()
flush()
seek(pos)
tell()
closed

















































