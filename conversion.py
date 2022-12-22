#!/usr/bin/env python
# coding: utf-8

# Inspired by DigitalSreeni YTChannel in Image Segmentation using U-Net
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib2 import Path
from keras.utils import array_to_img as array_to_img
from skimage.transform import resize


#  uniformo il datasert con la funzione normalize a 512x512
def _normalize(vol):
    hu_max = 256
    hu_min = -256
    vol = np.clip(vol, hu_min, hu_max)

    mxval = np.max(vol)
    mnval = np.min(vol)
    volume_norm = (vol - mnval) / max(mxval - mnval, 1e-3)

    return volume_norm


def _conversion_nii_to_data_array(path: Path):
    """
    Il path dovrà avere come terminazione un .nii.gz e restiturà l'array per keras

    :param path:
    :return: keras image obj
    """

    ct_Img = nib.load(path)
    ct_Img = ct_Img.get_fdata()
    ct_Img = _normalize(ct_Img)

    return ct_Img


def keras_array_of_cases(flag: int, start: int, end: int):
    """
    Questa funzione restituisce una lista np array che contiene tutti i cases (a loro volta keras img obj) , scelti da start a end
    Inizialmente entra nella directory e crea un primo array con il caso start , ne fa l'append e ripete per tutti i case fino a end


    :param flag: se 1 solo converte imaging, se 0 converte segmentation
    :param start: un int che indica da quale caso partire per il .append della lista
    :param end:(come start) dove arriva
    :return: np array finale
    """

    # inizializzo rispetto alla struttura del Dataset di kits19, all'interno del mio Drive
    PACKAGE_LOCATION = Path(os.path.abspath('drive/MyDrive/Colab Notebooks/kits19'))
    DATA_LOCATION = PACKAGE_LOCATION / "data"

    type = 'imaging'
    if flag == 0:
        type = 'segmentation'
    else:
        type = 'imaging'

    cases = []
    for out in range(start, end):
        # costituisco un array di un array
        print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
        print('Loading and normalizing case_' + f'{out:05}' + ' case...')

        temp_path = DATA_LOCATION / f'case_{out:05}' / (str(type) + '.nii.gz')

        #case corrisponde al numpy array di un solo case
        case =  _conversion_nii_to_data_array(temp_path)

        slice_factor = 15/100  #ex: su 100 slice  ne prendo 15
        slice_totali = int(case.shape[0])
        slice_ridotte = (slice_totali*(4/5)) - (slice_totali * (1/5)) # elimino 1/5 in alto e 1/5 in basso 

        passo = int(slice_ridotte * slice_factor)

        #per le ct con poche slice --> prendo tutte le slice : passo=1
        if passo==0:
          passo = 1

        print( 'Slice totali: ' + str(int(case.shape[0])) )


        #Inserisco una slice alla volta, cercando di prenderle in modo uniforme
        for j in range(int(slice_totali*(1/5)), int(slice_totali*(4/5))  ,passo):

          print("Caso : "+ str(out) +" slice scelta num: "+ str(j) + " su slice totali: " + str(int( case.shape[0])) + " con passo: " + str(passo) ) 
          cases.append(resize(case[j, :, :], (256, 256)))
        
    return np.array(cases)
