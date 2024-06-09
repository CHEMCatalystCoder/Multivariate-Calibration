function RMSE = LOOCV_OLS(X, y)
    n = size(X, 1);
    RMSE = 0;
    
    for i = 1:n
        X_train = X([1:i-1, i+1:end], :);
        y_train = y([1:i-1, i+1:end], :);
        
        B = inv(X_train' * X_train) * X_train' * y_train;
        y_pred = X(i, :) * B;
        
        RMSE = RMSE + (y_pred - y(i)).^2;
    end
    RMSE = sqrt(RMSE / n);
end