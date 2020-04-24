import os
import random
import shutil

real_digit_dir = './digit-data/real-digits'
syn_digit_dir = './digit-data/syn-digits'

if __name__ == '__main__':
    # make folders for train & test for 2 domains - only need to run once
    if not os.path.exists(os.path.join(real_digit_dir, 'train')):
       os.makedirs(os.path.join(real_digit_dir, 'train'))
    
    if not os.path.exists(os.path.join(real_digit_dir, 'test')):
        os.makedirs(os.path.join(real_digit_dir, 'test'))

    if not os.path.exists(os.path.join(syn_digit_dir, 'train')):
        os.makedirs(os.path.join(syn_digit_dir, 'train'))
    
    if not os.path.exists(os.path.join(syn_digit_dir, 'test')):
        os.makedirs(os.path.join(syn_digit_dir, 'test'))

    # copy 20% of files to test folders
    total_num_real = len(os.listdir(real_digit_dir))
    total_num_syn = len(os.listdir(syn_digit_dir))

    sample_test_real = random.sample(os.listdir(real_digit_dir), k=int( 1/5 * total_num_real))
    sample_test_syn = random.sample(os.listdir(syn_digit_dir), k=int( 1/5 * total_num_syn))

    for file in sample_test_real:
        shutil.move(os.path.join(real_digit_dir, file), os.path.join(real_digit_dir, 'test'))
    
    for file in sample_test_syn:
        shutil.move(os.path.join(syn_digit_dir, file), os.path.join(syn_digit_dir, 'test'))

    # move remaining files to train folders
    for file in os.listdir(real_digit_dir):
        if not os.path.isdir(os.path.join(real_digit_dir, file)): # ignore the test folder
            shutil.move(os.path.join(real_digit_dir, file), os.path.join(real_digit_dir, 'train'))
    
    for file in os.listdir(syn_digit_dir):
        if not os.path.isdir(os.path.join(syn_digit_dir, file)):
            shutil.move(os.path.join(syn_digit_dir, file), os.path.join(syn_digit_dir, 'train'))