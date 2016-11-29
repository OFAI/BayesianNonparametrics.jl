"Log PDF of Generalized student-t Distribution."
function tlogpdf(x, df, mean, sigma)

   function tdist_consts(df, sigma)
       hdf = 0.5 * df
       shdfhdim = hdf + 0.5
       v = lgamma(hdf + 1/2) - lgamma(hdf) - 0.5*log(df) - 0.5*log(pi) - log(sigma)
       return (shdfhdim, v)
   end

    shdfhdim, v = tdist_consts(df, sigma)

    xx = x .- mean
    xx = (xx ./ sqrt(sigma)).^2

    p = 1/df * xx

    return v - log((1 + p).^((df+1) / 2))
end
