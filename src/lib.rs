#[cfg(test)]
extern crate assert;

extern crate dwt;
extern crate probability;
extern crate statistics;

pub type Error = &'static str;
pub type Result<T> = std::result::Result<T, Error>;

macro_rules! raise(
    ($message:expr) => (return Err($message));
);

/// A beta model.
pub struct Beta {
    pub p: Vec<f64>,
    pub mu: f64,
    pub sd: f64,
}

impl Beta {
    /// Fit a multifractal wavelet model with beta-distributed multipliers.
    ///
    /// `m` is the minimal number of scaling coefficients at the coarsest level
    /// that should be used for the estimation of the statistics of the wavelet
    /// coefficients.
    pub fn fit(data: &[f64], m: usize) -> Result<Beta> {
        if m == 0 {
            raise!("`m` should be positive");
        }

        let n = data.len();
        let nscale = {
            let nscale = (n as f64 / m as f64).log2().floor();
            if nscale < 1.0 {
                raise!("`m` is too high (not enough data)");
            }
            nscale as usize
        };
        let ncoarse = (n as f64 / (1 << nscale) as f64).floor() as usize;

        let mut data = (&data[0..(ncoarse * (1 << nscale))]).to_vec();

        dwt::forward(&mut data, &dwt::wavelet::Haar::new(), nscale);

        let var_w = (0..nscale).map(|k| {
            let i = ncoarse * (1 << k);
            let j = ncoarse * (1 << (k + 1));
            mean_square(&data[i..j])
        }).collect::<Vec<_>>();

        let data = &data[0..ncoarse];

        let mu = statistics::mean(data);
        let sd = statistics::variance(data).sqrt();

        let mut p = Vec::with_capacity(nscale);
        p.push(0.5 * (mean_square(data) / var_w[0] - 1.0));
        if p[0] <= 0.0 {
            raise!("cannot fit the data");
        }

        for i in 0..(nscale - 1) {
            let eta = var_w[i] / var_w[i + 1];
            let pr = eta * 0.5 * (p[i] + 1.0) - 0.5;
            p.push(pr);
            if p[i + 1] <= 0.0 {
                raise!("cannot fit the data");
            }
        }

        Ok(Beta { p: p, mu: mu, sd: sd })
    }
}

#[inline]
fn mean_square(data: &[f64]) -> f64 {
    &data.iter().fold(0.0, |sum, &x| sum + x * x) / data.len() as f64
}

#[cfg(test)]
mod tests {
    use assert;
    use Beta;

    #[test]
    fn fit() {
        let data = [
            4.018080337519417e-01, 7.596669169084191e-02, 2.399161535536580e-01,
            1.233189348351655e-01, 1.839077882824167e-01, 2.399525256649028e-01,
            4.172670690843695e-01, 4.965443032574213e-02, 9.027161099152811e-01,
            9.447871897216460e-01, 4.908640924680799e-01, 4.892526384000189e-01,
            3.377194098213772e-01, 9.000538464176620e-01, 3.692467811202150e-01,
            1.112027552937874e-01, 7.802520683211379e-01, 3.897388369612534e-01,
            2.416912859138327e-01, 4.039121455881147e-01, 9.645452516838859e-02,
            1.319732926063351e-01, 9.420505907754851e-01, 9.561345402298023e-01,
            5.752085950784656e-01, 5.977954294715582e-02, 2.347799133724063e-01,
            3.531585712220711e-01, 8.211940401979591e-01, 1.540343765155505e-02,
            4.302380165780784e-02, 1.689900294627044e-01, 6.491154749564521e-01,
            7.317223856586703e-01, 6.477459631363067e-01, 4.509237064309449e-01,
            5.470088922863450e-01, 2.963208056077732e-01, 7.446928070741562e-01,
            1.889550150325445e-01, 6.867754333653150e-01, 1.835111557372697e-01,
        ];

        let model = Beta::fit(&data, 5).unwrap();

        assert::close(&model.p, &[
            1.635153583946054e+01, 2.793188701574629e+00, 3.739374677617142e+00,
        ], 1e-14);
        assert::close(&[model.mu], &[1.184252871226982e+00], 1e-14);
        assert::close(&[model.sd], &[4.466592147518644e-01], 1e-14);
    }
}
