import SimpleITK as sitk
import numpy as np
def get_spinedata(dataDir, pedicle_points, pedicle_points_in_zyx, input_z=, input_y=, input_x=):
    mask = sitk.ReadImage(dataDir)
    spacing = list(mask.GetSpacing())
    tmp = spacing[0]
    spacing[0] = spacing[2]
    spacing[2] = tmp
    mask_array = sitk.GetArrayFromImage(mask)
    mask_array, pedicle_points = resize(mask_array, pedicle_points, input_z, input_y, input_x)
    if pedicle_points_in_zyx: 
        temp = pedicle_points[:, 0].copy()
        pedicle_points[:, 0] = pedicle_points[:, 2]
        pedicle_points[:, 2] = temp
    mask_array = np.transpose(mask_array, (2, 1, 0))
    mask_coards = np.zeros((3,mask_array.shape[0],mask_array.shape[1],mask_array.shape[2]))
    for i in range(mask_coards[0].shape[0]):
        mask_coards[0][i,:,:] = i
    for i in range(mask_coards[1].shape[1]):
        mask_coards[1][:,i,:] = i 
    for i in range(mask_coards[2].shape[2]):
        mask_coards[2][:,:,i] = i
    return {"mask_coards":mask_coards, 
            "mask_array":mask_array, 
            "pedicle_points":pedicle_points,
            "spacing":spacing}

def resize(img, pts,input_s,input_h,input_w):
        s,h,w = img.shape
        pts[:, 0] = pts[:, 0]/s*input_s
        pts[:, 1] = pts[:, 1]/h*input_h
        pts[:, 2] = pts[:, 2]/w*input_w
        img = resize_image_itk(sitk.GetImageFromArray(img), newSize=[input_w,input_h,input_s], resamplemethod=sitk.sitkLinear)
        return sitk.GetArrayFromImage(img), pts


def resize_image_itk(itkimage, newSize, resamplemethod=sitk.sitkNearestNeighbor):
    resampler = sitk.ResampleImageFilter()
    origin = itkimage.GetOrigin()
    #print('origin: ', origin)
    originSize = itkimage.GetSize() 
    #print('originSize: ', originSize)
    originSpacing = itkimage.GetSpacing()
    #print('originSpacing: ',originSpacing)
    newSize = np.array(newSize, float)
    factor = originSize / newSize
    newSpacing = originSpacing * factor
    newSize = newSize.astype(np.int32) 
    resampler.SetReferenceImage(itkimage) 
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itkimgResampled = resampler.Execute(itkimage)  
    #itkimgResampled = itkimgResampled.SetSpacing([1, 1, 1])
    return itkimgResampled
