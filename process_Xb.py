from PIL import Image
import os

def process_syn_imgs(save_dir, img_filepath):
    img = Image.open(img_filepath)
    img_name = os.path.splitext(os.path.basename(img_filepath))[0]
    
    if not os.path.exists(os.path.join(save_dir, 'cropped')):
        os.makedirs(os.path.join(save_dir, 'cropped'))

    if not os.path.exists(os.path.join(save_dir, 'b1')):
        os.makedirs(os.path.join(save_dir, 'b1'))
    
    if not os.path.exists(os.path.join(save_dir, 'b2')):
        os.makedirs(os.path.join(save_dir, 'b2'))
    
    if not os.path.exists(os.path.join(save_dir, 'b3')):
        os.makedirs(os.path.join(save_dir, 'b3'))

    b2 = (0, 0, 32, 32)
    crop_b2 = img.crop(b2)
    crop_b2.save(save_dir + '/b1/b2_' + str(img_name) + '.png')

    b1 = (0, 32, 32, 64)
    crop_b1 = img.crop(b1)
    crop_b1.save(save_dir + '/b2/b1_' + str(img_name) + '.png')

    b3 = (0, 64, 32, 96)
    crop_b3 = img.crop(b3)
    crop_b3.save(save_dir + '/b3/b3_' + str(img_name) + '.png')


if __name__ == '__main__':
    syn_img_train_dir = './digit-data/syn-digits/train'
    syn_img_test_dir = './digit-data/syn-digits/test'

    train_imgs = [os.path.join(syn_img_train_dir, img_name) for img_name in os.listdir(syn_img_train_dir)]
    test_imgs = [os.path.join(syn_img_test_dir, img_name) for img_name in os.listdir(syn_img_test_dir)]

    for img_filepath in train_imgs:
        process_syn_imgs(syn_img_train_dir, img_filepath)
    
    for img_filepath in test_imgs:
        process_syn_imgs(syn_img_test_dir, img_filepath)
    