
# cython: language_level=3
# cython: infer_types=False
# cython: boundscheck=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: wraparound=False

from libc.math cimport sqrt
import numpy as np
cimport numpy as np
cimport cython
from cython.parallel import prange
from libc.stdint cimport int32_t
from scipy.linalg.cython_blas cimport daxpy,dgemm

 
cdef extern from "gsl/gsl_sf_legendre.h" nogil:
    void gsl_sf_legendre_Pl_array(int lmax, double x, double * result)

cdef extern from "gsl/gsl_sf_bessel.h" nogil:
    int gsl_sf_bessel_Jn_array(int nmin, int nmax, double x, double * result)
    int gsl_sf_bessel_In_array(int nmin, int nmax, double x, double * result)
    int gsl_sf_bessel_In_scaled_array(int nmin, int nmax, double x, double * result)
    int gsl_sf_bessel_yl_array(int lmax, double x, double * result)
    int gsl_sf_bessel_il_scaled_array(int lmax, double x, double * result)

cdef extern from "gsl/gsl_sf_gegenbauer.h" nogil:
    int gsl_sf_gegenpoly_array(int n, double lambda_val, double x, double * result)

cdef extern from "gsl/gsl_sf_hermite.h" nogil:
    int gsl_sf_hermite_array(int n, double x, double * result)
    


cdef void matrix_multiply_intermediate(double[:,:] a, double[:,:] b, double[:,:] c, double alpha=1.0, double beta=0.0) nogil except *:
    cdef:
        char *transa = 'n'
        char *transb = 'n'
        int m, n, k, lda, ldb, ldc
        double *a0 = &a[0,0]
        double *b0 = &b[0,0]
        double *c0 = &c[0,0]

    k = b.shape[0]
    m = b.shape[1]
    n = a.shape[0]

    dgemm(transa, transb, &m, &n, &k, &alpha, b0, &m, a0, &k, &beta, c0, &m)
    
    
cpdef np.ndarray[np.float64_t, ndim=2] filter_q0_norm(np.ndarray[np.float64_t, ndim=2] input_array, int threshold):
    cdef int num_rows = input_array.shape[0]
    cdef int num_columns = input_array.shape[1]
    cdef int count, i, j
    cdef list valid_rows = []

    # Iterate through each row
    for i in range(num_rows):
        count = 0
        for j in range(num_columns):
            if input_array[i, j] != 0.0:
                count += 1
        
        # If count does not surpass threshold, add the row to the list of valid rows
        if count <= threshold:
            valid_rows.append(input_array[i])

    # Convert the list of valid rows back to a numpy array
    return np.array(valid_rows, dtype=np.float64)    
       

cdef inline double compute_Lp_norm_for_array(double[:] orders, double p, double threshold, int size) nogil:
    cdef double norm = 0.0
    cdef double order_value
    cdef int i

    for i in range(size):
        order_value = orders[i]
        norm += order_value**p if order_value >= 0 else (-order_value)**p
        
    if 0 < p < 1:
        return norm <= threshold**p
    else:
        return (norm)**(1.0/p) <= threshold

cpdef np.ndarray[np.float64_t, ndim=2] filter_combinations(np.ndarray[np.float64_t, ndim=2] combinations, double p, double threshold):
    cdef int num_combinations = combinations.shape[0]
    cdef int num_values = combinations.shape[1]
    cdef list allowed_combinations = []    
    cdef int i
    
    for i in range(num_combinations):
        if compute_Lp_norm_for_array(combinations[i, :], p, threshold, num_values):
            allowed_combinations.append(combinations[i, :])
    
    return np.array(allowed_combinations)
    
cdef double fast_power(double base, int exp) nogil:
    cdef double result = 1.0
    while exp:
        if exp & 1:
            result *= base
        exp >>= 1
        base *= base
    return result    
 
cpdef np.ndarray[np.float64_t, ndim=2] calculate_legendre_expansion(np.ndarray[np.float64_t, ndim=2] allowed_combinations, 
                                                          np.ndarray[np.float64_t, ndim=2] values_2d, 
                                                          int order_max):
    cdef int num_combinations = allowed_combinations.shape[0]
    cdef int num_values = allowed_combinations.shape[1]
    cdef int num_rows = values_2d.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] expansion_2d = np.zeros((num_rows, num_combinations))
    cdef double[:, :] legendre_values = np.empty((num_values, order_max + 1))
    cdef double expansion_value
    cdef int order, i, j, val_idx

    # Outer loop for each row of the input 2D array
    for j in range(num_rows):
        

        # Calculate Legendre values for each value in the array
        for val_idx in range(num_values):
            gsl_sf_legendre_Pl_array(order_max,  values_2d[j, val_idx], &legendre_values[val_idx, 0])

        # Calculate expansion for the current row
        for i in range(num_combinations):
            expansion_value = 1.0
            for val_idx in range(num_values):
                order = int(allowed_combinations[i, val_idx])
                expansion_value *= legendre_values[val_idx, order]
            expansion_2d[j, i] = expansion_value

    return expansion_2d 
    
    
cpdef np.ndarray[np.float64_t, ndim=2] calculate_legendre_expansion_norm(np.ndarray[np.float64_t, ndim=2] allowed_combinations, 
                                                          np.ndarray[np.float64_t, ndim=2] values_2d, 
                                                          int order_max):
    cdef int num_combinations = allowed_combinations.shape[0]
    cdef int num_values = allowed_combinations.shape[1]
    cdef int num_rows = values_2d.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] expansion_2d = np.zeros((num_rows, num_combinations))
    cdef double[:, :] legendre_values = np.empty((num_values, order_max + 1))
    cdef double expansion_value, normalization_factor
    cdef int order, i, j, val_idx

    # Outer loop for each row of the input 2D array
    for j in range(num_rows):

        # Calculate Legendre values for each value in the array
        for val_idx in range(num_values):
            gsl_sf_legendre_Pl_array(order_max, values_2d[j, val_idx], &legendre_values[val_idx, 0])
            
            # Normalize each Legendre value
            for order in range(order_max + 1):
                normalization_factor = <double>(sqrt(2 * order + 1))
                legendre_values[val_idx, order] *= normalization_factor

        # Calculate expansion for the current row
        for i in range(num_combinations):
            expansion_value = 1.0
            for val_idx in range(num_values):
                order = int(allowed_combinations[i, val_idx])
                expansion_value *= legendre_values[val_idx, order]
            expansion_2d[j, i] = expansion_value

    return expansion_2d
    
cpdef np.ndarray[np.float64_t, ndim=2] calculate_legendre_expansion_openmp(np.ndarray[np.float64_t, ndim=2] allowed_combinations, 
                                                           np.ndarray[np.float64_t, ndim=2] values_2d, 
                                                           int order_max):
    cdef int num_combinations = allowed_combinations.shape[0]
    cdef int num_values = allowed_combinations.shape[1]
    cdef int num_rows = values_2d.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] expansion_2d = np.zeros((num_rows, num_combinations))
    cdef np.ndarray[np.float64_t, ndim=3] legendre_values_3d = np.empty((num_rows, num_values, order_max + 1))
    cdef np.ndarray[np.float64_t, ndim=2] expansion_values_2d = np.empty((num_rows, num_combinations)) 
    cdef int order, i, j, val_idx

    # Parallelize the outer loop using prange
    for j in prange(num_rows, nogil=True):

        # Calculate Legendre values for each value in the array
        for val_idx in range(num_values):
            gsl_sf_legendre_Pl_array(order_max, values_2d[j, val_idx], &legendre_values_3d[j, val_idx, 0])

        # Calculate expansion for the current row
        for i in range(num_combinations):
            expansion_values_2d[j, i] = 1.0
            for val_idx in range(num_values):
                order = int(allowed_combinations[i, val_idx])
                expansion_values_2d[j, i] *= legendre_values_3d[j, val_idx, order]
            
            expansion_2d[j, i] = expansion_values_2d[j, i]

    return expansion_2d    
    
 
cpdef np.ndarray[np.float64_t, ndim=2] calculate_legendre_expansion_norm_openmp(np.ndarray[np.float64_t, ndim=2] allowed_combinations, 
                                                          np.ndarray[np.float64_t, ndim=2] values_2d, 
                                                          int order_max):
    cdef int num_combinations = allowed_combinations.shape[0]
    cdef int num_values = allowed_combinations.shape[1]
    cdef int num_rows = values_2d.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] expansion_2d = np.zeros((num_rows, num_combinations))
    cdef double[:, :] legendre_values = np.empty((num_values, order_max + 1))
    cdef np.ndarray[np.float64_t, ndim=2] expansion_values_2d = np.empty((num_rows, num_combinations)) 
    cdef double expansion_value, normalization_factor
    cdef int order, i, j, val_idx

    # Outer loop for each row of the input 2D array
    for j in range(num_rows):

        # Calculate Legendre values for each value in the array
        for val_idx in range(num_values):
            gsl_sf_legendre_Pl_array(order_max, values_2d[j, val_idx], &legendre_values[val_idx, 0])
            
            # Normalize each Legendre value
            for order in range(order_max + 1):
                normalization_factor = <double>(sqrt(2 * order + 1))
                legendre_values[val_idx, order] *= normalization_factor

        # Calculate expansion for the current row
        for i in range(num_combinations):
            expansion_values_2d[j, i] = 1.0
            for val_idx in range(num_values):
                order = int(allowed_combinations[i, val_idx])
                expansion_values_2d[j, i] *= legendre_values[val_idx, order]
            expansion_2d[j, i] = expansion_values_2d[j, i]

    return expansion_2d
    
cpdef np.ndarray[np.float64_t, ndim=2] calculate_legendre_expansion_norm2(np.ndarray[np.float64_t, ndim=2] allowed_combinations, 
                                                          np.ndarray[np.float64_t, ndim=2] values_2d, 
                                                          int order_max):
    cdef int num_combinations = allowed_combinations.shape[0]
    cdef int num_values = allowed_combinations.shape[1]
    cdef int num_rows = values_2d.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] expansion_2d = np.zeros((num_rows, num_combinations))
    cdef double[:, :] legendre_values = np.empty((num_values, order_max + 1))
    cdef double expansion_value, normalization_factor
    cdef int order, i, j, val_idx

    # Outer loop for each row of the input 2D array
    for j in range(num_rows):

        # Calculate Legendre values for each value in the array
        for val_idx in range(num_values):
            gsl_sf_legendre_Pl_array(order_max, values_2d[j, val_idx], &legendre_values[val_idx, 0])
            
            # Normalize each Legendre value
            for order in range(order_max + 1):
                normalization_factor = <double>(sqrt(2 * order + 1))
                legendre_values[val_idx, order] *= 1.0/normalization_factor

        # Calculate expansion for the current row
        for i in range(num_combinations):
            expansion_value = 1.0
            for val_idx in range(num_values):
                order = int(allowed_combinations[i, val_idx])
                expansion_value *= legendre_values[val_idx, order]
            expansion_2d[j, i] = expansion_value

    return expansion_2d    
    
    
cpdef np.ndarray[np.float64_t, ndim=2] calculate_legendre_expansion_norm3(np.ndarray[np.float64_t, ndim=2] allowed_combinations, 
                                                          np.ndarray[np.float64_t, ndim=2] values_2d, 
                                                          int order_max):
    cdef int num_combinations = allowed_combinations.shape[0]
    cdef int num_values = allowed_combinations.shape[1]
    cdef int num_rows = values_2d.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] expansion_2d = np.zeros((num_rows, num_combinations))
    cdef double[:, :] legendre_values = np.empty((num_values, order_max + 1))
    cdef double expansion_value, normalization_factor
    cdef int order, i, j, val_idx

    # Outer loop for each row of the input 2D array
    for j in range(num_rows):

        # Calculate Legendre values for each value in the array
        for val_idx in range(num_values):
            gsl_sf_legendre_Pl_array(order_max, values_2d[j, val_idx], &legendre_values[val_idx, 0])
            
            # Normalize each Legendre value
            for order in range(order_max + 1):
                normalization_factor =<double>(sqrt(2 * order + 1)/sqrt(order + 1))
                legendre_values[val_idx, order] *= normalization_factor

        # Calculate expansion for the current row
        for i in range(num_combinations):
            expansion_value = 1.0
            for val_idx in range(num_values):
                order = int(allowed_combinations[i, val_idx])
                expansion_value *= legendre_values[val_idx, order]
            expansion_2d[j, i] = expansion_value

    return expansion_2d        
    
cpdef np.ndarray[np.float64_t, ndim=2] calculate_legendre_expansion_norm4(np.ndarray[np.float64_t, ndim=2] allowed_combinations, 
                                                          np.ndarray[np.float64_t, ndim=2] values_2d, 
                                                          int order_max):
    cdef int num_combinations = allowed_combinations.shape[0]
    cdef int num_values = allowed_combinations.shape[1]
    cdef int num_rows = values_2d.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] expansion_2d = np.zeros((num_rows, num_combinations))
    cdef double[:, :] legendre_values = np.empty((num_values, order_max + 1))
    cdef double expansion_value, normalization_factor
    cdef int order, i, j, val_idx

    # Outer loop for each row of the input 2D array
    for j in range(num_rows):

        # Calculate Legendre values for each value in the array
        for val_idx in range(num_values):
            gsl_sf_legendre_Pl_array(order_max, values_2d[j, val_idx], &legendre_values[val_idx, 0])
            
            # Normalize each Legendre value
            for order in range(order_max + 1):
                normalization_factor = <double>(sqrt(2 * order + 1)/sqrt(order + 1))
                legendre_values[val_idx, order] *= 1.0/normalization_factor

        # Calculate expansion for the current row
        for i in range(num_combinations):
            expansion_value = 1.0
            for val_idx in range(num_values):
                order = int(allowed_combinations[i, val_idx])
                expansion_value *= legendre_values[val_idx, order]
            expansion_2d[j, i] = expansion_value

    return expansion_2d          
    
    

cpdef np.ndarray[np.float64_t, ndim=2] calculate_power_expansion(np.ndarray[np.float64_t, ndim=2] allowed_combinations, 
                                                                  np.ndarray[np.float64_t, ndim=2] values_2d):
    cdef int num_combinations = allowed_combinations.shape[0]
    cdef int num_values = allowed_combinations.shape[1]
    cdef int num_rows = values_2d.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] expansion_2d = np.zeros((num_rows, num_combinations))
    cdef double expansion_value
    cdef int power, i, j, val_idx

    # Outer loop for each row of the input 2D array
    for j in range(num_rows):
        # Calculate expansion for the current row
        for i in range(num_combinations):
            expansion_value = 1.0
            for val_idx in range(num_values):
                power = int(allowed_combinations[i, val_idx])
                expansion_value *= fast_power(values_2d[j, val_idx], power) # values_2d[j, val_idx] ** power
            expansion_2d[j, i] = expansion_value

    return expansion_2d
    
cpdef np.ndarray[np.float64_t, ndim=2] calculate_power_expansion_openmp(np.ndarray[np.float64_t, ndim=2] allowed_combinations, 
                                                               np.ndarray[np.float64_t, ndim=2] values_2d):
    cdef int num_combinations = allowed_combinations.shape[0]
    cdef int num_values = allowed_combinations.shape[1]
    cdef int num_rows = values_2d.shape[0]
    cdef np.ndarray[np.float64_t, ndim=2] expansion_2d = np.zeros((num_rows, num_combinations))
    cdef int power, i, j, val_idx
    cdef np.ndarray[np.float64_t, ndim=2] expansion_value = np.empty((num_rows, num_combinations)) 

      
            
            # Outer loop for each row of the input 2D array
    for j in prange(num_rows, nogil=True):
        # Calculate expansion for the current row
        for i in range(num_combinations):
            expansion_value[j, i] = 1.0
            for val_idx in range(num_values):
                power = int(allowed_combinations[i, val_idx])
                expansion_value[j, i] *= fast_power(values_2d[j, val_idx], power)
            expansion_2d[j, i] = expansion_value[j, i]

    return expansion_2d    
    
    
cpdef predict_one_cosmo(double[:,:] yteste, double[:,:,:] Wtrans, double[:,:,:] A, double[:,:] b):
    cdef:
        int num_slices = Wtrans.shape[0]
        int intermediate_rows = yteste.shape[0]
        int intermediate_cols = Wtrans.shape[2]
        int final_rows = intermediate_rows
        int final_cols = A.shape[2]
        double[:,:] intermediate_result = np.empty((intermediate_rows, intermediate_cols))
        double[:,:] final_result_slice = np.empty((final_rows, final_cols))
        double[:,:,:] final_result = np.empty((num_slices, final_rows, final_cols))
        
        # Use daxpy to add vector b[i,:] to final_result_slice 
        cdef:
            int j,i
            int incx = 1
            int incy = 1
            double alpha = 1.0        

    for i in range(num_slices):
        # Compute yteste @ Wtrans[:,:]
        matrix_multiply_intermediate(yteste[:,:], Wtrans[i,:,:], intermediate_result)

        # Compute intermediate_result @ A[i,:,:]
        matrix_multiply_intermediate(intermediate_result, A[i,:,:], final_result_slice)
        
        for j in range(final_rows):
            daxpy(&final_cols, &alpha, &b[i,0], &incx, &final_result_slice[j, 0], &incy)
            
        # Store the result in final_result[i,:,:]
        final_result[i,:,:] = final_result_slice

    return np.asarray(final_result) 
 
cpdef predict_cosmos(double[:,:,:] yteste, double[:,:,:] Wtrans, double[:,:,:] A, double[:,:] b):
    cdef:
        int num_slices = Wtrans.shape[0]
        int intermediate_rows = yteste.shape[1]
        int intermediate_cols = Wtrans.shape[2]
        int final_rows = intermediate_rows
        int final_cols = A.shape[2]
        double[:,:] intermediate_result = np.empty((intermediate_rows, intermediate_cols))
        double[:,:] final_result_slice = np.empty((final_rows, final_cols))
        double[:,:,:] final_result = np.empty((num_slices, final_rows, final_cols))
        
        # Use daxpy to add vector b[i,:] to final_result_slice 
        cdef:
            int j,i
            int incx = 1
            int incy = 1
            double alpha = 1.0        

    for i in range(num_slices):
        # Compute yteste @ Wtrans[i,:,:]
        matrix_multiply_intermediate(yteste[i,:,:], Wtrans[i,:,:], intermediate_result)

        # Compute intermediate_result @ A[i,:,:]
        matrix_multiply_intermediate(intermediate_result, A[i,:,:], final_result_slice)
        
        for j in range(final_rows):
            daxpy(&final_cols, &alpha, &b[i,0], &incx, &final_result_slice[j, 0], &incy)
            
        # Store the result in final_result[i,:,:]
        final_result[i,:,:] = final_result_slice

    return np.asarray(final_result)
    
 
    
    
cpdef predict_cosmos_openmp(double[:,:,:] yteste, double[:,:,:] Wtrans, double[:,:,:] A, double[:,:] b):
    cdef:
        int num_slices = Wtrans.shape[0]
        int intermediate_rows = yteste.shape[0]
        int intermediate_cols = Wtrans.shape[2]
        int final_rows = intermediate_rows
        int final_cols = A.shape[2]
        double[:,:] intermediate_result = np.empty((intermediate_rows, intermediate_cols))
        double[:,:] final_result_slice = np.empty((final_rows, final_cols))
        double[:,:,:] final_result = np.empty((num_slices, final_rows, final_cols))
        
        # Use daxpy to add vector b[i,:] to final_result_slice 
        cdef:
            int j,i
            int incx = 1
            int incy = 1
            double alpha = 1.0        
 
    for i in prange(num_slices, nogil=True):
        # Compute yteste @ Wtrans[:,:]
        matrix_multiply_intermediate(yteste[i,:,:], Wtrans[i,:,:], intermediate_result)

        # Compute intermediate_result @ A[i,:,:]
        matrix_multiply_intermediate(intermediate_result, A[i,:,:], final_result_slice)
        
        for j in range(final_rows):
            daxpy(&final_cols, &alpha, &b[i,0], &incx, &final_result_slice[j, 0], &incy)
            
        # Store the result in final_result[i,:,:]
        final_result[i,:,:] = final_result_slice

    return np.asarray(final_result)
    
    
cpdef predict_one_cosmo_openmp(double[:,:] yteste, double[:,:,:] Wtrans, double[:,:,:] A, double[:,:] b):
    cdef:
        int num_slices = Wtrans.shape[0]
        int intermediate_rows = yteste.shape[0]
        int intermediate_cols = Wtrans.shape[2]
        int final_rows = intermediate_rows
        int final_cols = A.shape[2]
        double[:,:] intermediate_result = np.empty((intermediate_rows, intermediate_cols))
        double[:,:] final_result_slice = np.empty((final_rows, final_cols))
        double[:,:,:] final_result = np.empty((num_slices, final_rows, final_cols))
        
        # Use daxpy to add vector b[i,:] to final_result_slice 
        cdef:
            int j,i
            int incx = 1
            int incy = 1
            double alpha = 1.0        

    for i in prange(num_slices, nogil=True):
        # Compute yteste @ Wtrans[:,:]
        matrix_multiply_intermediate(yteste[:,:], Wtrans[i,:,:], intermediate_result)

        # Compute intermediate_result @ A[i,:,:]
        matrix_multiply_intermediate(intermediate_result, A[i,:,:], final_result_slice)
        
        for j in range(final_rows):
            daxpy(&final_cols, &alpha, &b[i,0], &incx, &final_result_slice[j, 0], &incy)
            
        # Store the result in final_result[i,:,:]
        final_result[i,:,:] = final_result_slice

    return np.asarray(final_result)     
    
    
    
