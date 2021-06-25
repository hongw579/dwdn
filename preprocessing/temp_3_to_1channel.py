import numpy as np

blur_dir = 'data_3channel/InputBlurredImage/train_blur.npy'
kernel_dir = 'data_3channel/psfMotionKernel/train_kernel.npy'
true_dir = 'data_3channel/InputTargetImage/train_true.npy'

blur = np.load(blur_dir)
kernel = np.load(kernel_dir)
true = np.load(true_dir)

blur = np.transpose(blur, axes=(0,3,1,2))
kernel = np.transpose(kernel, axes=(0,3,1,2))
true = np.transpose(true, axes=(0,3,1,2))

blur = np.reshape(blur, (24000, 35, 35))
kernel = np.reshape(kernel, (24000, 35, 35))
true = np.reshape(true, (24000, 35, 35))

blur = blur[:, 1:33, 1:33]
#kernel = kernel[:, 1:33, 1:33]
true = true[:, 1:33, 1:33]

print(blur.shape, kernel.shape, true.shape)

np.save('./data_1channel/psfMotionKernel/train_kernel.npy',kernel[..., np.newaxis])
np.save('./data_1channel/InputTargetImage/train_true.npy', true[..., np.newaxis])
np.save('./data_1channel/InputBlurredImage/train_blur.npy', blur[..., np.newaxis])

blur_dir = 'data_3channel/TestData/blurredImage/test_blur.npy'
kernel_dir = 'data_3channel/TestData/kernelImage/test_kernel.npy'
true_dir = 'data_3channel/TestData/GTImage/test_true.npy'

blur = np.load(blur_dir)
kernel = np.load(kernel_dir)
true = np.load(true_dir)

blur = np.transpose(blur, axes=(0,3,1,2))
kernel = np.transpose(kernel, axes=(0,3,1,2))
true = np.transpose(true, axes=(0,3,1,2))

print(blur.shape, kernel.shape, true.shape)

blur = np.reshape(blur, (6000, 35, 35))
kernel = np.reshape(kernel, (6000, 35, 35))
true = np.reshape(true, (6000, 35, 35))

blur = blur[:, 1:33, 1:33]
#kernel = kernel[:, 1:33, 1:33]
true = true[:, 1:33, 1:33]

print(blur.shape, kernel.shape, true.shape)

np.save('./data_1channel/TestData/kernelImage/test_kernel.npy', kernel[..., np.newaxis])
np.save('./data_1channel/TestData/GTImage/test_true.npy', true[..., np.newaxis])
np.save('./data_1channel/TestData/blurredImage/test_blur.npy', blur[..., np.newaxis])

