import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image_all = np.load('test_10000.npz')

kernel = image_all['arr_0'][:,0,:,:]
true = image_all['arr_0'][:,2,:,:]
blur = image_all['arr_0'][:,1,:,:]

blur = blur[:, :, 1:33, 1:33]
true = true[:, :, 1:33, 1:33]

blur_max = np.absolute(blur).max(axis=-1).max(axis=-1).max(axis=-1)
true_max = np.absolute(true).max(axis=-1).max(axis=-1).max(axis=-1)
maximum = np.maximum(blur_max, true_max)

blur_input = blur/maximum[:,None, None, None]
true_input = true/maximum[:,None, None, None]

blur_input = np.transpose(blur_input, axes=(0,2,3,1))
true_input = np.transpose(true_input, axes=(0,2,3,1))
kernel_input = np.transpose(kernel, axes=(0,2,3,1))

np.save('./data_3channel/psfMotionKernel/train_kernel.npy',kernel_input[:8000])
np.save('./data_3channel/InputTargetImage/train_true.npy',true_input[:8000])
np.save('./data_3channel/InputBlurredImage/train_blur.npy',blur_input[:8000])

np.save('./data_3channel/TestData/psfMotionKernel/test_kernel.npy',kernel_input[8000:])
np.save('./data_3channel/TestData/InputTargetImage/test_true.npy',true_input[8000:])
np.save('./data_3channel/TestData/InputBlurredImage/test_blur.npy',blur_input[8000:])
