try:
    from svm_script import *
except:
    from scripts.svm_script import *

if __name__ == '__main__':
    
    svm_path = '/home/withme/dev/tensorflow-for-poets-2/svm_long/'
    out_dir_svm = '/home/withme/dev/tensorflow-for-poets-2/svm_80_percent_train_summary/'

    data_path = '/home/withme/dev/tensorflow-for-poets-2/svm_long/svmdata_2018-06-12 17:11:42.126471.p'

    find_best_model_load(data_path, svm_path, out_dir_svm)