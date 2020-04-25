# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:39:11 2020

This is my answer for the question in https://qr.ae/pNvTdB . 
There’s an array of heads of linked lists that may or may not intersect with each other. 
The goal is to group the array entries such that if two linked lists intersect 
with each other then they belong to the same group.

Advantage with this code is, this doesn't ;oop till tail to find the grp.
Once it finds that the link is chaned to another linked list, the head is
immediately added to corresponding group.
@author: Nirmal
"""

# define the array of linked list as dictionary of dictionaries

arr_ll = { 0: {0:'a','a':'b','b':'c','c':None}
, 1: {1:'d','d':'b','b':'c','c':None}
, 2: {2:'e','e':'f','f':'g','g':'c','c':None}
, 3: {3:'h','h':None}
, 4: {4:'i','i':'j','j':None}
, 5: {5:'l','l':'c','c':None}
, 6: {6:'k','k':'j','j':None}
, 7: {7:None}}

out = dict()                        # output dict of groups

# function to add head of linked list to corresponding group.
# if no group is fo

def add(g,h):
  if g in out:
    out[g].append(h)
  else:
    out[g] = [h]

grp = 1                             # group number

# this dict maintains which grp does each node belongs to
traversed = dict()                

for head, ll in arr_ll.items():
  node = head
  while True:
    next = ll[node]                 
    if not next:                    # if tail node
      add(grp, head)                # add head to group
      grp+=1                        # increment grp number count
      break
    if next in traversed:           # if link already seen before
      add(traversed[next], head)    # add head to already existing grp
      break
    else:                           # new link never seen before
      traversed[next] = grp         # add link to new grp
      node = next

print(out)