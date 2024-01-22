import torch
import torchvision
from torch.utils.data import Dataset
import lmdb
import pickle
from lmdb_image import FFHQ_image
from torchvision import transforms
from PIL import Image
import io
import h5py
from PIL import Image
import os.path
import string
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets.vision import VisionDataset
from collections.abc import Iterable
from typing import Any, Callable, cast, List, Optional, Tuple, Union
from torchvision.datasets.utils import iterable_to_str, verify_str_arg
#lsun bedroom val path to lmdb file: /home/usuaris/imatge/claudia.giardina/imageSynthesis
#lsun bedroom train path to lmdb file: /mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/data_for_tests/
#ffhq lmdb: /mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/data_for_tests/ffhq_lmdb
class LSUNClass(VisionDataset):
    def __init__(self, root, transform = None, target_transform = None, size=256, flip_p=0.5):
        transform = transforms.Compose([
                  transforms.Resize(size),
                  transforms.RandomHorizontalFlip(flip_p),
                  transforms.CenterCrop(size),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "_cache_" + "".join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index: int):
        example = {'relative_file_path_': '/mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/data_for_tests/ffhq_lmdb', 
                    'file_path_': '/mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/data_for_tests/ffhq_lmdb'}
        img = None
        target = torch.tensor([1])
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        example["image"] = img
        return example

    def __len__(self) -> int:
        return self.length


class LSUNBedroomsTrain(LSUNClass):
    def __init__(self, **kwargs):
        super().__init__(root='/mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/data_for_tests/bedroom_train_lmdb',  **kwargs)


class LSUNBedroomsValidation(LSUNClass):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(root='/home/usuaris/imatge/claudia.giardina/imageSynthesis/bedroom_val_lmdb', flip_p=flip_p, **kwargs)
      
class FFHQ_lmdb(Dataset):
    def __init__(self,  size=256, flip_p=0.5):
        self.path_data = '/mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/data_for_tests/ffhq_lmdb' 
        self.transform = transforms.Compose([
                  transforms.Resize(size),
                  transforms.RandomHorizontalFlip(flip_p),
                  transforms.CenterCrop(size),
                  transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)])
        # Delay loading LMDB data until after initialization to avoid "can't pickle Environment Object error"
        self.env = lmdb.open(self.path_data, subdir=os.path.isdir(self.path_data),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        self.env = None
        

    def _init_db(self):
        #print('env initialization')
        self.env = lmdb.open(self.path_data, subdir=os.path.isdir(self.path_data),
            readonly=True, lock=False,
            readahead=False, meminit=False)
        self.txn = self.env.begin()
        #print(f"len of dataset in init function {self.env.stat()['entries']}")
        #self.len_dataset = self.env.stat()['entries']

    def read_lmdb(self, image_id):
        # Encode the key the same way as we stored it
        lmdb_data = self.txn.get(f"{image_id:0>7}".encode("ascii"))
        #a FFHQ_image object is loaded
        lmdb_data = pickle.loads(lmdb_data)
        return lmdb_data

    def __getitem__(self, index):
        """ Read a single image from LMDB.
            Parameters:
            ---------------
            image_id    integer unique ID for image

            Returns:
            ----------
            tensor       image of (3, 256, 256)
        """
        
        example = {'relative_file_path_': '/mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/data_for_tests/ffhq_lmdb', 
                    'file_path_': '/mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/data_for_tests/ffhq_lmdb'}
        # Delay loading LMDB data until after initialization
        if self.env is None:
            self._init_db()
        #print(f"len of dataset: {self.env.stat()['entries']}")  
        #reatrieve the relevant bits
        image = self.read_lmdb(index)    
        image = image.get_image()
        #image = copy(image) #to make writable 
        example["image"] = self.transform(Image.fromarray(np.uint8(image)).convert('RGB'))

        return example
        
    def __len__(self):
        return self.length

class PICAI_slices(Dataset):
    def __init__(self, root, size=256, flip_p=0.5):
        """
        if mode == 'train':
            self.path_data = '/mnt/gpid08/datasets/FLUTE/PICAI/picai_slices_train.h5' 
        else:
            self.path_data = '/mnt/gpid08/datasets/FLUTE/PICAI/picai_slices_val.h5' 
        """
        self.path_data = root
        self.augmentation = False
        self.modality = 't2w'
        self.transform = transforms.Compose([
                  #transforms.Resize(size),
                  #transforms.RandomHorizontalFlip(flip_p),
                  #transforms.CenterCrop(size),
                  transforms.ToTensor(),
                  transforms.Normalize((204.36695600614223, 204.36695600614223, 204.36695600614223), (145.4163783588631, 145.4163783588631, 145.4163783588631), inplace=True)])
        if self.augmentation and self.mode != 'test':
            # Flip saggital
            transform = transforms.Compose([
                #transforms.Normalize([0], [1]),
            ])
            self.transform = transform
        self.dataset = None
        with h5py.File(self.path_data, 'r') as file:
            self.dataset_len = len(file[f"/{self.modality}"])
        """
        train_list, test_list = train_test_split(list(range(0,dataset_len)), train_size = 0.9, shuffle = True, random_state=0)
        if mode == 'train':
            self.dataset_len = len(train_list)
        else:
            self.dataset_len = len(test_list)
        """

    def __getitem__(self, index):
        #open HDF5 file once to avoid overhead
        if self.dataset is None:
            #logger.info('opening hdf5 file once...')
            self.dataset = h5py.File(self.path_data, 'r')
            """
            We'll use this to bypass the slow h5py data access with a faster memory mapping (only works on uncompressed contiguous datasets):
            """
            self.ds_mri = self.dataset[f"/{self.modality}"]
            # We get the dataset address in the HDF5 file.
            offset_mri = self.ds_mri.id.get_offset()
            # We ensure we have a non-compressed contiguous array.
            assert self.ds_mri.chunks is None
            assert self.ds_mri.compression is None
            assert offset_mri > 0
            dtype_mri = self.ds_mri.dtype
            shape_mri = self.ds_mri.shape

            #np.memmap: Create a memory-map to an array stored in a binary file on disk.
            self.arr_mri = np.memmap(self.path_data, mode='r', shape=shape_mri, offset=offset_mri, dtype=dtype_mri)

        example = {'relative_file_path_': '/mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/PICAI', 
                    'file_path_': '/mnt/gpid08/users/claudia.giardina/ImageSynthesis3D/PICAI'}
        slice_i = np.array(self.arr_mri[index]).astype("float32")
        #slice_i = Image.fromarray(slice_i).convert('RGB') # converting to PIL image and RGB
        #copy same info into the three channels
        slice_i = slice_i.reshape(1,slice_i.shape[0],slice_i.shape[1])
        example['original_slice'] = slice_i
        slice_i = np.repeat(slice_i,3,axis=0)
        #slice_i = self.transform(slice_i.transpose(1,2,0)) #to tensor, and other transforms if are included
        #example['image'] = slice_i.numpy()
        example['image'] = (slice_i - slice_i.mean())/slice_i.std() #instance z-score normalization

        return example

    def __len__(self):
        return self.dataset_len

class PICAISlicesTrain(PICAI_slices):
    def __init__(self, **kwargs):
        super().__init__(root='/mnt/gpid08/datasets/FLUTE/PICAI/picai_slices.h5',  **kwargs)
        #super().__init__(root='/mnt/gpid08/datasets/FLUTE/PICAI/picai_slices_train.h5',  **kwargs)


class PICAISlicesValidation(PICAI_slices):
    def __init__(self, flip_p=0.0, **kwargs):
        super().__init__(root='/mnt/gpid08/datasets/FLUTE/PICAI/picai_slices_val_1500.h5', flip_p=flip_p, **kwargs)
        #super().__init__(root='/mnt/gpid08/datasets/FLUTE/PICAI/picai_slices_val.h5', flip_p=flip_p, **kwargs)
