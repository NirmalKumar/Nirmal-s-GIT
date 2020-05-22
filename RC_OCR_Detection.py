# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:56:15 2020

This program employs tesseract OCR to read vehicle RC cards, collect data and 
write them to an excel file.

OpenCV opens the image, grayscales it, converts to pure black and white image.
Tesseract runs the ocr and creates text from processed image.
Various functions uses regular expressions to find data: 
    Registration Number, Name of the owner, Chassis number and Engine Number
from the text.
These details are stored in dictionary for each image and then stored in 
excel file using DataFrames.

@author: Nirmal
"""

# import required packages
import os
import cv2
import pytesseract
import numpy as np
import re
from pandas import DataFrame as df
import time

start = time.time()                 # to measure time elapsed
regn_dict = {'Image Name' : [],     # dict to store data from image
             'Regn No.' : [],
             'Name of owner' : [],
             'Chassis No.' : [],
             'Engine No.' : []}

all_text = []                       # this stores ocr text from each image
nan = np.NaN

###########################################################################
# function to find registration number
def find_regnno_opt1(regn, regn_phrase_found):
    '''

    Parameters
    ----------
    regn : String
        input text from which registration number is searched
    regn_phrase_found : Boolean
        False: Registration Number label not yet found
        True: Registration Number label is already found
        
    Logic
    -----
    First, search for regn related phrases. 
    If found and more characters are present in same line, 
        separate it as registration number and return data
    Else
        Return only registration phrase is found.
    
    If registration phrase is already found
        search for AA----NNNN pattern and return ot as registration number
        
    Returns
    -------
    Registration Number 
    boolean whether registration number phrase found or not
    
    Sample
    ------
    INPUT: REGN . NO-: DL5CJ 4987
    OUTPUT: DL5CJ4987

    '''
    if ('REGN' in regn and 'NO' in regn) or ('REGISTRATION NO' in regn):
        if regn.find('REGN')>2:     #regn should start at beginning of text
            return None, False
        
        regn = regn.replace('REGN','')      # remove regn related words
        regn = regn.replace('REGISTRATION','')
        regn = regn.replace('NO','')
        if len(regn) > 9:                   # if still text has enough chars
            regn = re.sub(r'\W+', '', regn) # remove all non-alphanumerics
            return regn, True               # treat it as registration number
        else:
            return None, True               # say regn related label found
        
    elif regn_phrase_found == True:         # if reg no text already found
        if len(regn) > 9:
            regn = re.sub(r'\W+', '', regn) # remove all non-alphanumerics
            return regn, True               # treat it as registration number
    return None, False

###########################################################################
# function to find registration number if regn no related text is not found
    
def find_regnno_opt2(regn):
    '''  

    Parameters
    ----------
    regn : string
        Text from which registration number is searched
        
    Logic
    -----
    Match every text with this pattern: AA----NNNN
    If matches, then return it as registration number.
    Use this apprach as last resort if registration number is not found in
    previous function

    Returns
    -------
    Registration Number string
    
    Sample
    ------
    INPUT: 33GN . NO-: DL5CJ 4987
    OUTPUT: DL5CJ4987
    '''

    regn = re.sub(r'\W+', '', regn)
    if ((8 < len(regn) < 12) and re.match('(([A-Z]{2})[\w]+([\d]{4}))', regn)):
        return regn
    return None
    
###########################################################################
# function to find name of the owner  
def find_name(name, name_phrase_found):
    '''

    Parameters
    ----------
    name : string
        Text from which name of owner is searched
    name_phrase_found : boolean
        False: Name label not yet found
        True: Name label is already found
        
    Logic
    -----
    First search for NAME field. If found, remove associated text
    Remaining text should be name of owner

    Returns
    -------
    Name of owner: String
    boolean whether NAME text found or not
    
    Sample
    ------
    INPUT: CRAMER'S NAME MR VINAY KUMAR HANDA
    OUTPUT: MR VINAY KUMAR HANDA
    

    '''
    if 'NAME' in name:
        loc = name.find('NAME')
        if loc > 0:
            name = name[loc:]               # get past Name Label
        
        name = name.replace('NAME','')
        name = name.replace('ADDRESS','')   # Remove associated label text
        name = name.replace('OWNER','')     # like address, owner, etc.
        name = name.replace('AND','')
        name = name.replace('OF','')
        name = re.findall(r'[A-Z\s\.]+', name)  # Name can contain only alpha, space and .
        if len(name) > 0 and len(name[0]) > 5:  # Name shd be atleast 5 chars
            name = ''.join(name)            
            name = name.strip()
            return name, True               # return name
        else:
            return None, True               # say NAME label found
        
    elif name_phrase_found == True:         # if NAME label already found
        if len(name) > 5:
            name = re.findall(r'[A-Z\s\.]+', name)
            name = ''.join(name)
            name = name.strip()
            return name, True               # return name
        else:
            return None, True
    return None, False

###########################################################################
# function to find chassis number
    
def find_chassis(chassis):
    '''

    Parameters
    ----------
    chassis : string
        Text from which chassis number is searched

    Logic
    -----
    Match text with MA|B-----NNNNN and length between 16 and 19
    If ok, return it as chassis number
    
    Returns
    -------
    Chassis number : String
    
    Sample
    ------
    INPUT: CH. NO : MA3EWDE1S00526615
    OUTPUT: MA3EWDE1S00526615

    '''
    chassis = re.sub(r'\W+', '', chassis)       # Remove all non-alphanumerics    
    chassis = re.findall(r'([M][AB]\w+[\d]{3})', chassis)   # pattern match?
    if len(chassis) > 0 and (14 < len(chassis[0]) < 19):                  # ideal length of chassis no is 17
        if chassis != []:
            return chassis[0]
    return None

###########################################################################
# function to find engine number
def find_engine(engine, engine_phrase_found):
    '''

    Parameters
    ----------
    engine : string
        Text from which engine number is searched
    name_phrase_found : boolean
        False: Engine label not yet found
        True: Engine label is already found

    Logic
    -----
    Search for Engine in text. If found, remove associated labels.
    If it has more chars, then ot shd be engine number. 
    Else
        Search for ANN----NNNN pattern in further lines.
    
    Returns
    -------
    Engine number : String
    boolean whether Engine text found or not
    
    Sample
    ------
    INPUT: ENGINE = KT2MN 1530003
    OUTPUT: KT2MN1530003

    '''
    if 'ENO' in engine or 'E NO' in engine or 'ENGINE' in engine:
        engine = engine.replace('ENGINE','')            # Remove engine related labels
        engine = engine.replace('ENO','')
        engine = engine.replace('E NO','')
        engine = engine.replace('NO','')
        engine = re.sub(r'\W+', '', engine)             # Remove all non-alphanumerics
        if 10 < len(engine) < 14:                       # ideal length of engine no is 12
            return engine, True
        else:
            return None, True
    elif engine_phrase_found == True:
        engine = re.sub(r'\W+', '', engine)             # Remove all non-alphanumerics
        engine = re.findall(r'([A-Z]{1}[\d]{2}[A-Z]{1}\w+[\d]{4})', engine) # pattern match?
        if len(engine) > 0 and (10 < len(engine[0]) < 14):  # ideal length of engine no is 12
            if engine != []:
                return engine[0],True
            else:
                return None, True
        else:
            return None, True
    return None, False

'''
    Main process starts here. 
    Step 1: Open the image. Resize it to length 1024 pixels
    Step 2: Turn image to Grayscale
    Step 3: Apply threshold to make it pure black and white
    Step 4: Run Tesseract OCR engine to get text from the processed image
'''	
image_folder = 'RC'
files = os.listdir(image_folder)
path = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
pytesseract.pytesseract.tesseract_cmd = path

for image_name in files:                    # for each file in the folder
    if image_name.endswith('.jpg'):         # check if it is jpg image
        img = cv2.imread(image_folder + '\\' + image_name)  # open image using opencv
        if img.shape[1] > 1024:             # for bigger images, resize it to length 1024 pixels
            ratio = 1024/img.shape[1]
            img = cv2.resize(img, None, fx=ratio, fy=ratio)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     # change to grayscale image
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 15)
                                            # all non-black pixels to become white
        config = "--psm 12"                 # tesseract configuration value - Sparse text with OSD
        text = pytesseract.image_to_string(img, config=config, lang="eng")  #run ocr and get text from image
        regn_dict['Image Name'].append(image_name)      # add Image Name to dictionary
        text = text.upper()                 # convert whole text to upper case
        text = text.split('\n\n')           # turn text to array of individual lines
        all_text.append(text)               # store text for test purpose
		
        '''
        From here, call functions to find Registration Number, Name of owner, etc.
        Each result, whether foun or not is stored in regn_dict dictionary.
        '''

# find registration no	
        '''
        Search for Registration text and Registration Number in each line.
        If both are found in same line, happy and get out of loop.
        If only regn related label is found and no number is found, search for regn number in next line.
        '''

        lines_elapsed = 0			
        phrase = False
        for line in text:
            regnno, phrase = find_regnno_opt1(line, phrase) # go find the registration number
            if phrase == True:                              # if label found but regn number not found
                lines_elapsed += 1                          # search for regn no in next line
            if regnno != None or lines_elapsed > 1:         # if regn no found or not found in both lines
                break
        
        if regnno == None:                          # if regn no not found in first attempt,
            for line in text:
                regnno = find_regnno_opt2(line)     # use second option
                if regnno != None:
                    break
                
        if regnno != None:                          # store result in dictionary
            regn_dict['Regn No.'].append(regnno)
        else:
            regn_dict['Regn No.'].append(nan)
            
# find name of owner            
        '''
        Search for Name text and Name in each line.
        If both are found in same line, happy and get out of loop.
        If only Name label is found and no name is found, search for name in next line.
        '''			
        
        lines_elapsed = 0			
        phrase = False
        for line in text:
            name, phrase = find_name(line, phrase)
            if phrase == True:
                lines_elapsed += 1
            if name != None or lines_elapsed > 1:
                break
                
        if name != None:                            # store result in dictionary
            regn_dict['Name of owner'].append(name)
        else:
            regn_dict['Name of owner'].append(nan)

# find chassis number	
        '''
        Search for chassis number in each line.
        Chassis number pattern is used in the function.
        '''		
        	
        for line in text:
            chassis = find_chassis(line)
            if chassis != None:
                break

        if chassis != None:                         # store result in dictionary
            regn_dict['Chassis No.'].append(chassis)
        else:
            regn_dict['Chassis No.'].append(nan)
			
# find engine number
        '''
        Search for Engine related text and Engine Number in each line.
        If both are found in same line, happy and get out of loop.
        If only Engine related label is found and no engine number is found, 
        search for it in next 6 lines.
        Sometimes, Engine number is far down from ENO like this:

        ENO             <-- Text for engine
        CH.NO : MA3FDEB{200381622 -
        COLOUR
        S. SILVER
        D13A1833010     <-- Actual Engine Number
        CLASS
        '''		
        lines_elapsed = 0
        phrase = False            
        for line in text:
            engine, phrase = find_engine(line, phrase)
            if phrase == True:
                lines_elapsed += 1
            if engine != None or lines_elapsed >= 6:
                break
        
        if engine != None:
            regn_dict['Engine No.'].append(engine)
        else:
            regn_dict['Engine No.'].append(nan)
            
'''
Phew, we finished them all. Lets put them in an excel for our view.
'''
regn_df = df.from_dict(regn_dict)                   # convert dictionary to dataframe
regn_df.to_excel('regn_dets.xls', index = False)    # write to excel file

total_time = time.time() - start                    # find total time elapsed