import numpy as np

def tflc_beamformer(Y_in, mask, a_vec=None, n_beamformers=2, iterations=20):
    X = np.moveaxis(Y_in, 0, -1).astype(np.complex64)
    F_dim, T_dim, M = X.shape
    
    # Initialize steering vectors if not provided
    if a_vec is None:
        a_vec = np.ones((F_dim, M), dtype=np.complex64)
    else:
        a_vec = a_vec.astype(np.complex64)
        
    noise_mask = (1.0 - mask).astype(np.float32)
    debug_data = {}
    
    # 1. Estimate Global Noise Covariance Matrix via vectorized Einstein summation
    # Replaces the nested loops over f, m1, m2 for lightning-fast execution
    Phi_noise_total = np.einsum('ft,ftm,ftn->fmn', noise_mask, X, np.conj(X))
    debug_data['Phi_init'] = Phi_noise_total.copy()
    
    # 2. Initialize Weight Matrix W_k: shape (F_dim, M, n_beamformers)
    W_k = np.zeros((F_dim, M, n_beamformers), dtype=np.complex64)
    perturbation = 0.01 * (np.ones((M, M)) + 1j * np.ones((M, M)))
    
    for k in range(n_beamformers):
        for f in range(F_dim):
            # Regularize the matrix to guarantee invertibility (diagonal loading)
            Phi_f = Phi_noise_total[f] + perturbation + 1e-2 * np.eye(M)
            
            try:
                Phi_inv = np.linalg.inv(Phi_f)
                a = a_vec[f]
                num = Phi_inv @ a
                den = np.conj(a) @ num
                W_k[f, :, k] = num / (den + 1e-10)
            except np.linalg.LinAlgError:
                # Fallback to standard Delay-and-Sum weight if matrix is singular
                W_k[f, :, k] = a_vec[f] / M
                
    debug_data['W_init'] = W_k.copy()
    
    # 3. Iterative Optimization Loop
    for it in range(iterations):
        # Calculate intermediate output for each beamformer path
        Y_k = np.einsum('fmk,ftm->ftk', W_k, X)
        
        y1 = Y_k[:, :, 0]
        y2 = Y_k[:, :, 1]
        y21 = y1 - y2
        
        # Calculate soft-assignment spatial masks
        numerator = -np.real(y2 * np.conj(y21))
        denominator = np.abs(y21)**2 + 1e-10
        
        c1 = np.clip(numerator / denominator, 0.0, 1.0)
        c2 = 1.0 - c1
        c_k = np.stack([c1, c2], axis=-1)
        
        # Update beamformer covariance and re-optimize weights
        for k in range(n_beamformers):
            mask_k = c_k[:, :, k]
            Phi_k = np.einsum('ft,ftm,ftn->fmn', mask_k, X, np.conj(X))
            
            for f in range(F_dim):
                Phi_f = Phi_k[f] + 1e-2 * np.eye(M)
                try:
                    Phi_inv = np.linalg.inv(Phi_f)
                    a = a_vec[f]
                    W_k[f, :, k] = (Phi_inv @ a) / (np.conj(a) @ Phi_inv @ a + 1e-10)
                except np.linalg.LinAlgError:
                    pass # Retain previous valid weights if inversion fails
                    
    debug_data['W_final'] = W_k
    debug_data['c_k'] = c_k
    
    # Compute final combined beamformer output
    Y_k_final = np.einsum('fmk,ftm->ftk', W_k, X)
    Y_final = np.sum(c_k * Y_k_final, axis=2)
    
    return Y_final, debug_data