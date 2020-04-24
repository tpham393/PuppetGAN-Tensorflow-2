import tarfile

def extract(tar_file, save_path):
    opened_tar = tarfile.open(tar_file)
     
    if tarfile.is_tarfile(tar_file):
        opened_tar.extractall(save_path)
    else:
        print("The tar file you entered is not a tar file")

if __name__ == '__main__':
    extract('../data/real-digits.tar', './digit-data/real-digits')
    extract('../data/syn-digits.tar', './digit-data/syn-digits')