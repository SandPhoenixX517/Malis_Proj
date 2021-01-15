import os
# Setup data for three cities which are Columbus, Potsdam, Selwyn. In order to do the same for the other three cities just copy paste the code and modify the files names and the URLs
os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/detection/COWC_Detection_Columbus_CSUAV_AFRL.tbz")

os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/detection/COWC_Detection_Potsdam_ISPRS.tbz")

os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/detection/COWC_Detection_Selwyn_LINZ.tbz")

os.system("tar -xvjf ./COWC_Detection_Columbus_CSUAV_AFRL.tbz")

os.system("tar -xvjf ./COWC_Detection_Potsdam_ISPRS.tbz")

os.system("tar -xvjf ./COWC_Detection_Selwyn_LINZ.tbz")

os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/detection/COWC_test_list_detection.txt.bz2")
os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/detection/COWC_train_list_detection.txt.bz2")
os.system("bzip2 -d COWC_test_list_detection.txt.bz2")
os.system("bzip2 -d COWC_train_list_detection.txt.bz2")
os.system("cp COWC_test_list_detection.txt ./Columbus_CSUAV_AFRL/")
os.system("cp COWC_test_list_detection.txt ./Potsdam_ISPRS/")
os.system("cp COWC_test_list_detection.txt ./Selwyn_LINZ/")
os.system("cp COWC_train_list_detection.txt ./Selwyn_LINZ/")
os.system("cp COWC_train_list_detection.txt ./Potsdam_ISPRS/")
os.system("cp COWC_train_list_detection.txt ./Columbus_CSUAV_AFRL/")
