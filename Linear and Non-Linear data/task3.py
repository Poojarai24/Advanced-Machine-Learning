from sklearn.svm import SVC
from data import generate_xor_data
from visual import plot_2d_data
from sklearn.metrics import accuracy_score

def main():
    
    X,y = generate_xor_data(n = 200)
    plot_2d_data(X,y,title =" XOR Data (Original)")
    
    #Linear SVM
    linear_svm = SVC(kernel='linear')
    linear_svm.fit(X,y)
    y_pred_linear = linear_svm.predict(X)
    plot_2d_data(X,y_pred_linear,title="Linear SVM Predictions (Fails)")
    
    #Polynomial kernel SVM
    poly_svm = SVC(kernel='poly', degree=2)
    poly_svm.fit(X,y)
    y_pred_poly = poly_svm.predict(X)
    plot_2d_data(X,y_pred_poly,title="Polynomial Kernel SVM Predictions (Succeeds)")
    
    # RBF Kernel SVM
    rbf_svm = SVC(kernel = 'rbf', gamma='scale')
    rbf_svm.fit(X,y)
    y_pred_rbf = rbf_svm.predict(X)
    plot_2d_data(X,y_pred_rbf,title="RBF Kernel SVM Predictions (Succeds)")
    
    #try with two more kernel
    
    #print accuracy
    print("Linear SVM Accuracy: ", accuracy_score(y,y_pred_linear))
    print("Polynomial Kernel SVM Accuracy: ", accuracy_score(y,y_pred_linear))
    print("RBF Kernel SVM Accuracy: ", accuracy_score(y,y_pred_linear))

if __name__=="__main__":
    main()