from numpy.fft import fftn, ifftn, fftshift
import numpy as np
import imageio # used only for read the image

SAVE_COUNTER = 0

######
# METHODS CODE
######
def gaussian_filter(k, sigma):
	arx = np.arange((-k // 2) + 1.0, (k // 2) + 1.0)
	x, y = np.meshgrid(arx, arx)
	filt = np.exp( -(1/2)*(np.square(x) + np.square(y))/np.square(sigma) )
	return filt/np.sum(filt)

def ref_padding(matrix,ref):
	pad = (ref.shape[0]//2) - matrix.shape[0]//2
	matrix = np.pad(matrix, (pad,pad-1), "constant", constant_values=(0))
	return matrix

def denoise_img(img,filter_,save=False):
	denoised_img = np.multiply(fftn(img),fftn(filter_))
	denoised_img = np.real(fftshift(ifftn(denoised_img)))

	if save:
		global SAVE_COUNTER
		imageio.imwrite('plots/'+str(SAVE_COUNTER)+'_denoised_img.png',denoised_img.astype(np.uint8))
		SAVE_COUNTER += 1

	return denoised_img

def normalise_img(img, min_v, max_v, save=False):
	abs_img = np.abs(img)
	width, height = img.shape
	normalised_img = np.zeros((width,height))

	min_i = np.min(np.abs(img))
	max_i = np.max(np.abs(img))
	for x in range(width):
		for y in range(height):
			pixel = abs_img[x,y]
			normalised_img[x,y] = \
				(pixel-min_i)*((max_v-min_v)/(max_i-min_i)) + min_v

	if save:
		global SAVE_COUNTER
		imageio.imwrite('plots/'+str(SAVE_COUNTER)+'_normalised_img.png',normalised_img.astype(np.uint8))
		SAVE_COUNTER += 1

	return normalised_img

def laplacian_filter():
	filt = np.array([[ 0,-1, 0],
				  [-1, 4,-1],
				  [ 0,-1, 0]])
	return filt

def clsf(G,H,gamma):
	G = fftn(G)
	H = fftn(H)

	p = laplacian_filter()
	P = fftn(ref_padding(p,H))
	
	D = np.divide(np.conj(H),(np.real(H)**2 + gamma*(np.real(P)**2)))
	R = np.multiply(D,G)

	return np.real(fftshift(ifftn(R)))

######
# MAIN CODE
######
# 1. Reading the inputs and defining the variables
# a. image filename, k (filter size, a positive and odd integer),
# sigma (standard deviation, a float point number > 0 and <=1) and
# gamma (regularization factor, a float point number >= 0 and <=1)
filename = str(input()).rstrip()
k = int(input())
sigma = float(input())
gamma = float(input())

# b. loading the image
F = imageio.imread(filename)

# c. Create the gaussian filter 
H = gaussian_filter(k,sigma)
H = ref_padding(H,F)

# 2. Denosing the image
G_denoised = denoise_img(F,H)

# 3. Normalising the denoised image
G_normalised_n = normalise_img(G_denoised,0,np.max(F))

# 4. Debluring the image via the CLS method
G_deblur = clsf(G_denoised,H,gamma)

# 5. Normalising the image
G_normalised_b = normalise_img(G_deblur,0,np.max(G_normalised_n))

# 6. Printing the result
print('%.1f' % G_normalised_b.std())