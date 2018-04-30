'''
Created on Mar 10, 2018

@author: voldemaro
'''
import numpy as np;
import numpy.linalg as nla;


def extract_house_reflection(_A, betas, side="lower"):
    
    if (side == "lower"):
        A, shift = (_A, 0);
       
    else:
        A, shift = (_A.T,1);
    

    m,n = A.shape

    Q = np.eye(m, dtype=np.complex128);
    for j in reversed(xrange(shift, min(m,n))):
        if(side == "lower"):
            v = A[j:,j-shift].copy();
            v[0] = 1.0+0.0j;
            miniQ = np.eye(m-j, dtype=np.complex128) - betas[j-shift] * np.outer(v,np.asmatrix(v).getH());
            Q[j:,j:] = (miniQ).dot(Q[j:,j:]);
        else:
            v = A[j:,j-shift].copy();
            v[0] = 1.0+0.0j;
            miniQ = np.eye(m-j, dtype=np.complex128) - betas[j-shift] * np.outer(np.asmatrix(v).getH(),v);
            Q[j:,j:] = Q[j:,j:].dot(miniQ);
    return Q;

def extract_upper_bidiag(M):
    '''from M, pull out the diagonal and superdiagonal 
       like np.triu or np.tril would  ... assume rows >= cols
       (works for non-square M)'''
    B = np.zeros_like(M, dtype=np.complex128);

    shape = B.shape[1]
    step = shape + 1 # # cols + 1
    end  = shape**2  # top square (shape,shape)
    
    B.flat[ :end:step] = np.diag(M)     # diag
    B.flat[1:end:step] = np.diag(M,+1)  # super diag
    return B;

def extract_packed_house_bidiag(H, betas_U, betas_V):
    U  = extract_house_reflection(H, betas_U, side="lower")
    Vt = extract_house_reflection(H, betas_V, side="upper")
    B  = extract_upper_bidiag(H)
    return U, B, Vt





def make_house_vec_2(x):
    n = x.shape[0];
 
    # v is our return vector; we hack on v[0]
    v = np.copy(x);
    v_plus = np.copy(x);
    v_minus = np.copy(x);
 
    #getting the complex theta==q 
    q = nla.norm(x[0]);
    q = x[0]/nla.norm(x[0]);
    
    norm_x= nla.norm(x);
    
    v_plus[0] = x[0] + q * norm_x;
    v_minus[0] = x[0] - q * norm_x;
    
    norm_plus = nla.norm(v_plus);
    norm_minus = nla.norm(v_minus);

    if nla.norm(x[1:]) < np.finfo(float).eps*10.0: 
        beta = 0.0  ;
        v[0] = 1.0 + 0.0j;
    else:
        if (norm_plus < norm_minus):
            v = v_plus;
        else:
            v = v_minus;
    
            v = v / v[0];        
            beta = 2.0 / (nla.norm(v)**2);  
       
    return v, beta;

def house_bidiag(A):
    m,n = A.shape
    assert m >= n
    U,V = np.eye(m, dtype=np.complex128), np.eye(n, dtype=np.complex128);
    
    betas_U = np.empty(n, dtype=np.complex128);
    betas_V = np.empty(n-1, dtype=np.complex128);
    for j in xrange(n):
        
        u,betas_U[j] = make_house_vec_2(A[j:,j])
        miniHouse = np.eye(m-j, dtype=np.complex128) - betas_U[j] * np.outer(u,np.asmatrix(u).getH());
        A[j:,j:] = (miniHouse).dot(A[j:,j:])
        # A's rows go 0 to m
        #  A[j,j] and A[j,j+1] are part of the bidiagonal result
        #  A[j+1:,j]  (remainder of this COLUMN) stores u
        #  A[j+1:] means indices j+1, j+2, .... m-1 is m-j positions
        A[j+1:,j] = u[1:] # [1:m-j+1]

        if j < n-1:
            v,betas_V[j] = make_house_vec_2(A[j,j+1:].T)
            miniHouse = np.eye(n-(j+1), dtype=np.complex128) - betas_V[j] * np.outer(np.asmatrix(v).getH(),v);
            A[j:,j+1:] = A[j:, j+1:].dot(miniHouse)
            A[j ,j+2:] = v[1:] # [1:n-j]
    return betas_U, betas_V;


def main():    
    arraySize = 5;    
    filename = "random_array_5x5_complex2.dat";
    fileobj = open(filename, mode='rb'); 
    c2 = np.fromfile(filename, dtype = np.complex128);
    fileobj.close;
    
    c2 = np.reshape(c2, (-1, arraySize));  
    
    A_test = np.copy(c2)
        
    
    betas_U, betas_V = house_bidiag(c2);    
    U, B, Vt = extract_packed_house_bidiag(c2, betas_U, betas_V);
    
    print "B:\n", B;
    print "U:\n", U;
    print "Vt:\n", Vt;
    print "should equal input:", np.allclose(U.dot(B).dot(Vt), A_test);
    
if __name__ == '__main__':
    main();
