import os 

os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Columbus_CSUAV_AFRL.tbz")
os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Potsdam_ISPRS.tbz")

# this are for the other images.
#os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Selwyn_LINZ.tbz")
#os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Toronto_ISPRS.tbz")
#os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Vaihingen_ISPRS.tbz")
#os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_Counting_Utah_AGRC.tbz")
os.system("tar -xvjf ./COWC_Counting_Columbus_CSUAV_AFRL.tbz")
os.system("tar -xvjf ./COWC_Counting_Potsdam_ISPRS.tbz")
os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_test_list_64_class.txt.bz2")
os.system("wget https://gdo152.llnl.gov/cowc/download/cowc/datasets/patch_sets/counting/COWC_train_list_64_class.txt.bz2")
os.system("bzip2 -d COWC_test_list_64_class.txt.bz2")
os.system("bzip2 -d COWC_train_list_64_class.txt.bz2")
os.system("cp COWC_test_list_64_class.txt ./Columbus_CSUAV_AFRL/")
os.system("cp COWC_train_list_64_class.txt ./Potsdam_ISPRS/")
os.system("cp COWC_train_list_64_class.txt ./Columbus_CSUAV_AFRL/")
os.system("cp COWC_test_list_64_class.txt ./Potsdam_ISPRS/")
