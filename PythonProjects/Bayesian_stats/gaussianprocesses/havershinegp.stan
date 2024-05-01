functions
{
    vector radians(vector degrees) {
        return degrees * pi() / 180.0;
    }

    // Function to calculate the Haversine distance between two points given their latitude and longitude
    matrix haversine_distance(matrix X) {
        real R = 6.371; // Earth's radius in thoussuands kilometers
    
        int N = rows(X);
        matrix[N, 2] radians_X;
        matrix[N, N] latd;
        matrix[N, N] lond;
        matrix[N, N] d;
        matrix[N, N] a;
        matrix[N, N] c;

        
        radians_X[:, 1] = radians(X[:, 1]);
        radians_X[:, 2] = radians(X[:, 2]);
        

        for (i in 1:N) {
        for (j in 1:N) {
            latd[i, j] = radians_X[i, 1] - radians_X[j, 1];
            lond[i, j] = radians_X[i, 2] - radians_X[j, 2];
        }
        }

        for (i in 1:N) {
        for (j in 1:N) {
            d[i, j] = cos(radians_X[i, 1]) * cos(radians_X[j, 1]);
            a[i, j] = sin(latd[i, j] / 2)^2 + d[i, j] * sin(lond[i, j] / 2)^2;
            c[i, j] = 2 * atan2(sqrt(a[i, j]), sqrt(1 - a[i, j]));
        }
        }

        return R * c;
    }


    matrix cov_GPL2(matrix x, real sq_alpha, real lengthscale, real delta) {

        int N = dims(x)[1];
        matrix[N,N] dist_X = haversine_distance(x);
        matrix[N, N] K;
        for (i in 1:(N-1)) {
          K[i, i] = sq_alpha + delta;
          for (j in (i + 1):N) {
            K[i, j] =  sq_alpha * exp(-lengthscale * square(dist_X[i,j]) );
            K[j, i] = K[i, j];
          }
        }
        K[N, N] = sq_alpha + delta;
        return K;
        }
}
data
{
int N;
int k;
matrix[N, k] x;
vector[N] log_pop;
array[N] int<lower=0> y;
real alpha_prior;
real beta_prior;
}
transformed data
{
    print(haversine_distance(x));
}
parameters
{
real alpha;
real beta;
real<lower =0> eta;
real<lower =0> lengthscale_f;
vector[N] f;
}
model
{
     
    vector[N] lambda;
    matrix[N, N] SIGMA;

    //Priors
    alpha ~ normal(0, 5);
    beta ~ normal(0,1);
    eta ~ exponential(2);
    lengthscale_f ~ inv_gamma(alpha_prior, beta_prior);

    SIGMA = cov_GPL2(x, eta, lengthscale_f, 1e-12);
    f ~ multi_normal( rep_vector(0,N), SIGMA );

    lambda = alpha + beta * log_pop + f;

    // Likelihood
    y ~ poisson_log(lambda);
}