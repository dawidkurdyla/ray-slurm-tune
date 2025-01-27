import shutil
import os
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--source",required=True,type=str,help="Source path")
    parser.add_argument( "--dest",required=True,type=str,help="The second path.")

    # Parse the command-line arguments
    args = parser.parse_args()
    source_dir = args.source
    root_dir = args.dest
    train_dir = os.path.join(root_dir,'asl_train')
    test_dir = os.path.join(root_dir,'asl_test')

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    for class_ in os.listdir(source_dir):
        source_folder = os.path.join(source_dir,class_)
        dest_folder = os.path.join(train_dir,class_)
        os.mkdir(dest_folder)
        
        file_list = os.listdir(source_folder)
        ceil_95 = int(len(file_list) * .95)
        for file in file_list[:ceil_95]:
            shutil.copy2(os.path.join(source_folder,file),os.path.join(dest_folder,file))
        dest_folder_test = os.path.join(test_dir,class_)
        os.mkdir(dest_folder_test)
        for file in file_list[ceil_95:]:
            shutil.copy2(os.path.join(source_folder,file),os.path.join(dest_folder_test,file))
    