from torch.utils.data import Dataset
from preprocess import *

import matplotlib.image as mpimg

def store_patch_coords(path, patch_side, stride):
    with open(os.path.join(path, 'all.txt')) as f:
        lines = f.readline()
    names = lines.split(',')
    
    patch_coords = []
    for im_name in names:
        im_name = im_name.strip()
        n_path = path + im_name

        name = n_path.split('/')[-1]
        cm = mpimg.imread(n_path + '/cm/'+name+'-cm.tif')
        cm = cm - 1
        cm = np.expand_dims(cm,axis=2)
        cm = reshape_for_torch(cm)
        cm = get_padded_image(cm,patch_side,stride)
        
        s = cm.shape

        n1 = ceil((s[1] - patch_side + 1) / stride)
        n2 = ceil((s[2] - patch_side + 1) / stride)
        
        # generate path coordinates
        for i in range(n1):
            for j in range(n2):
                current_patch_coords = (im_name,  
                                [stride*i, stride*i + patch_side, stride*j, stride*j + patch_side],
                                [stride*(i + 1), stride*(j + 1)])
                
                # check is it worth
                limits = current_patch_coords[1]
                label = cm[:, limits[0]:limits[1], limits[2]:limits[3]]
                val, counts = np.unique(label, return_counts=True)

                if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 5% useful area with labels that are not 0
                    patch_coords.append(current_patch_coords)
                    
    for patch in patch_coords:
        print(patch)
    print(len(patch_coords)) 

class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(self, 
                 path, 
                 fname = 'all.txt',
                 bands = ['B01','B02','B03','B04','B05',
                                        'B06','B07','B08','B8A','B09',
                                        'B10','B11','B12'],
                 patch_side = 256, 
                 stride = 128, 
                 transform = None,
                 normalize = True, 
                 fp_modifier_for_weights=10.0):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        # basics
        self.transform = transform
        self.normalize = normalize
        self.path = path
        self.patch_side = patch_side
        if not stride:
            self.stride = 1
        else:
            self.stride = stride
        
#         print(path + fname)
        with open(os.path.join(path, fname)) as f:
            lines = f.readline()
        names = lines.split(',')
        self.names = names
        self.n_imgs = len(self.names)
        
        n_pix = 0
        true_pix = 0
        
        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        # print(f'Dataset: {self.names}')
        for im_name in self.names:
            im_name = im_name.strip()
            # load and store each image
            print(f'reading....{im_name}')

            I1, I2, cm = read_sentinel_img_trio(self.path + im_name,
                                                band_list=bands,
                                                patch_size=patch_side,
                                                stride=stride)
            self.imgs_1[im_name] = I1
            self.imgs_2[im_name] = I2
            self.change_maps[im_name] = cm
            
            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()

            n1 = ceil((s[1] - patch_side + 1) / stride)
            n2 = ceil((s[2] - patch_side + 1) / stride)

            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i
            
            # generate path coordinates
            for i in range(n1):
                for j in range(n2):

                    current_patch_coords = (im_name,  
                                    [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
                    print(current_patch_coords)

                    self.patch_coords.append(current_patch_coords)
                    
        self.weights = [ fp_modifier_for_weights * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        label = self.change_maps[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        sample = {'I1': I1, 'I2': I2, 'label': label, 'fname': im_name}
        
        if self.normalize:
            I1 = (I1 - I1.mean()) / I1.std()
            I2 = (I2 - I2.mean()) / I2.std()
        
        if self.transform:
            sample = self.transform(sample)

        return sample
        
    def get_weights(self):
        return self.weights
    
if __name__ == '__main__':
    
    PATH_TO_TRAIN_DATA = '../../../../DataSet/OSCD/'
    PATH_TO_TEST_DATA = '../../../../DataSet/OSCD/'
    PATH_TO_VAL_DATA = '../../../../DataSet/OSCD/'
    # BANDS = ['B01', 'B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12']
    BANDS = ['B01', 'B02']
    
    PATCH_SIDE = 256
    STRIDE = 128

    # train_dataset = ChangeDetectionDataset(path=PATH_TO_TRAIN_DATA,
    #                                     fname = 'train.txt', 
    #                                     patch_side = PATCH_SIDE, 
    #                                     stride = STRIDE,
    #                                     transform = None,
    #                                     bands=BANDS)
    # print(len(train_dataset))
    
    store_patch_coords(PATH_TO_TRAIN_DATA, PATCH_SIDE, STRIDE)