from DeepCSMRI 		import * # All the import were added in there
from GenerateMask 	import * # Import the mask generation for 2D image
##################################################################

def generatePair(images):
	"""
	Input is an 5D tensor images (1,1,dimz,dimy,dimx)
	Call the generateMask to make the single mask (dimy, dimx)
	Expand the mask along z and expand it dimension to (1,1,dimz,dimy,dimx)
	Make undersampled images consist of 3 channels (1,3,dimz,dimy,dimx)
		images[:,0,:,:,:] is the real part of zero filling
		images[:,1,:,:,:] is the imag part of zero filling
		images[:,2,:,:,:] is the mask for undersampling
	Make the prediction contains 256 level of
	"""
	
	# Get the shape of images
	# print images.shape
	dimn, dimz, dimy, dimx = images.shape
	shape = images.shape

	# print shape
	srcImage = np.zeros(shape)
	#srcImage = np.zeros((  1,dimz,dimy,dimx))
	dstImage = np.zeros(shape)
	###########################################################
	# Generate the 3 channel input
	# Generate the undersampling pattern
	mask   =  generateMask(dimn*dimz, dimy, dimx, sampling_rate=0.25, center_ratio=0.5)
	mask   =  np.reshape(mask, (dimn, dimz, dimy, dimx))
	# print "mask.shape"
	# print mask.shape
	# Perform forward Fourier transform
	kspace = np.fft.fft2(images)
	
	# Perform undersampling
	under  = mask*kspace
	del kspace
	# Perform inverse Fourier transform for zerofilling
	zfill  = np.fft.ifft2(under)
	del under
	# March through the temporal dimension
	#for z in range(dimz):
	# Assign the channels

	# srcImage = np.abs(zfill)  
	srcImage = np.real(zfill)
	
	srcImage = srcImage/np.max(srcImage) * 255.0
	# srcImage = np.expand_dims(srcImage, axis=0)
	
	#print srcImage.shape
	dstImage = images 
	
	###########################################################

 
	return srcImage, dstImage, mask
def test_generatePair(images):
	print "Here"
	print images.shape
	Xz, Xf, R = generatePair(images)
	print "After Generating"
	# print images.shape
	sliceId = 0
	full = Xf[0,sliceId,:,:]
	zero = Xz[0,sliceId,:,:]
	#tmp = 
	#tmp = y[sliceId,:,:]
	print full.shape
	print zero.shape
	# tmp = images
	plt.imshow( full, cmap=cm.gray) 
	plt.axis('off')
	plt.show()
	plt.imshow( zero, cmap=cm.gray) 
	plt.axis('off')
	plt.show()
##################################################################
if __name__ == '__main__':
	images = np.load('images.npy')

	# # Extract a single image
	# image  = images[0,0,:,:]
	# print image.shape
	# plt.imshow( np.squeeze(image) , cmap=cm.gray) 
	# plt.axis('off')
	# plt.show()
	test_generatePair(images[0:1,:,:,:])