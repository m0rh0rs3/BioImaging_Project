# BioImaging_Project
A computer vision project for NecstCAMP NL1

by
Morcavallo Pietro


**STEP 1**
Come prima cosa ho analizzato la struttura del dataset e la tipologia di file .nii, utilizzando MANGO (un visualizer per CT) con il quale ho notato la diversità fra i vari case: non tutti hanno lo stesso numero di slice.

**STEP 2**
Ho implementato una serie di funzioni per la conversione da .nii a formato compatibile con la rete neurale (vedi keras_array_of_cases() ), con la quale effettuo anche il directory management

**STEP 3**
Mi sono accorto che per il multiclass segmentation l'encoding preferibile da utilizzare è quello one_hot_encoding, così ho implementato questo tipo di conversione

**STEP 4**
A questo punto ispirandomi ad un altro progetto su la costruzione generale di una UNET , ho sviluppato la unet(), la sua costruzione e il setting degli hyperparametri
(219 - Understanding U-Net architecture and building it from scratch -DigitalSreeni on YouTube)

**STEP 5** 
Ho deciso di utilizzare una 10 di imagini per caso tagliate in modo uniforme cercando di prendere la parte centrale, eliminado quindi 1/5 in alto e 1/5 in basso dove generalmente nella mask vi è solo background

**STEP 6**
Per verificare le predizioni graficamente ho implementato, grazie a mathplotlib, delle sottofunzioni che mostrano contemporaneamente la CT del paziente test, la mask reale ,la mask predetta

NB: La rete ad oggi ancora non riesce a segmentare in modo corretto i reni, penso sia dovuto ad un problema di pochi dati.
