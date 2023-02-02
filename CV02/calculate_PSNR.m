
function PSNR =calculate_PSNR(f,F)
    GL = 255;  
    MSE=mean2((double(f)-double(F)).^2);
    PSNR= 10*log10( (GL*GL)/MSE );
end   
   