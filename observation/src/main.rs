use std::error::Error;
use std::fs::File;
use std::io::prelude::*;

use rand::Rng;
use rand_distr::{Distribution, Normal};

const N_OBS_STATION: usize = 40;
const N_TRUTH_TIME: usize = 2001;
const N_STEP_OBS: usize = 10;
const N_OBS_TIME: usize = 201;

#[derive(Clone)]
struct Truth {
    time: Vec<f64>,
    data: Vec<Vec<f64>>,
}

type DataResult<T> = Result<T, Box<dyn Error>>;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = "data/".to_string();
    let truth = read_raw_truth(&(base.clone() + "x_t_original.txt"))?;
    let truth_sync = sync_obs(truth.clone());
    let err_var = 0.2;

    let mean = raw_truth_mean(&truth);
    let sd = raw_truth_sd(&truth);
    println!("mean = {}, sd = {}", mean, sd,);
    let full_obs = full_process(truth.clone(), err_var);
    let full_obs = sync_obs(full_obs);
    let truth = common_process(truth, err_var);

    let sync_normal = sync_obs(truth.clone());
    let async_normal = async_obs(truth.clone());
    let sync_missing = missing_obs(sync_normal.clone());
    let async_missing = missing_obs(async_normal.clone());
    let sync_error = error_obs(sync_normal.clone(), err_var);
    let async_error = error_obs(async_normal.clone(), err_var);

    write_to_txt(&(base.clone() + "y_o_sync.txt"), sync_normal)?;
    write_to_txt(&(base.clone() + "y_o_async.txt"), async_normal)?;
    write_to_txt(&(base.clone() + "y_o_sync_miss.txt"), sync_missing)?;
    write_to_txt(&(base.clone() + "y_o_async_miss.txt"), async_missing)?;
    write_to_txt(&(base.clone() + "y_o_sync_err.txt"), sync_error)?;
    write_to_txt(&(base.clone() + "y_o_async_err.txt"), async_error)?;
    write_to_txt(&(base.clone() + "y_o_full.txt"), full_obs)?;
    write_truth(&(base.clone() + "x_t_sync.txt"), truth_sync)?;

    Ok(())
}

fn read_raw_truth(file: &str) -> DataResult<Truth> {
    let mut t = Vec::new();
    let mut d = Vec::new();
    let mut f = File::open(file)?;
    let mut buffer = String::new();

    f.read_to_string(&mut buffer)?;
    let data: Vec<&str> = buffer.trim_end().split(char::is_whitespace).collect();
    let tnd: Vec<f64> = data.into_iter().map(|s| s.parse().unwrap()).collect();
    // t for every 41, d for remain, 1t+40d
    let iter = tnd.chunks_exact(N_OBS_STATION + 1);
    for tnd in iter {
        let t_in = tnd.first().unwrap().to_owned();
        let d_in: Vec<f64> = tnd.iter().cloned().skip(1).take(N_OBS_STATION).collect();
        t.push(t_in);
        d.push(d_in);
    }

    Ok(Truth { time: t, data: d })
}

fn raw_truth_mean(truth: &Truth) -> f64 {
    let count = truth.data.iter().flatten().count();
    let sum: f64 = truth.data.iter().flatten().sum();
    sum / count as f64
}

fn raw_truth_sd(truth: &Truth) -> f64 {
    let mean = raw_truth_mean(truth);
    let count = truth.data.iter().flatten().count();
    (truth
        .data
        .iter()
        .flatten()
        .map(|x| (x - mean).powi(2))
        .sum::<f64>()
        / count as f64)
        .sqrt()
}

fn full_process(truth: Truth, sd: f64) -> Truth {
    let normal = Normal::new(0.0, sd).unwrap();
    let mut rng = rand::thread_rng();
    let data: Vec<Vec<f64>> = truth
        .data
        .into_iter()
        .map(|l| {
            let err_tmp = normal.sample_iter(&mut rng);
            l.into_iter()
                .enumerate()
                .zip(err_tmp)
                .map(|((_, d), e)| d + e)
                .collect()
        })
        .collect();

    Truth { data, ..truth }
}

fn common_process(truth: Truth, sd: f64) -> Truth {
    let normal = Normal::new(0.0, sd).unwrap();
    let mut rng = rand::thread_rng();
    let masked_data: Vec<Vec<f64>> = truth
        .data
        .into_iter()
        .map(|l| {
            let err_tmp = normal.sample_iter(&mut rng);
            l.into_iter()
                .enumerate()
                .zip(err_tmp)
                .map(|((i, d), e)| match i % 2 == 0 {
                    true => d + e,
                    false => f64::NAN,
                })
                .collect()
        })
        .collect();

    Truth {
        data: masked_data,
        ..truth
    }
}

fn sync_obs(truth: Truth) -> Truth {
    let time = truth.time.into_iter().step_by(N_STEP_OBS).collect();
    let data = truth.data.into_iter().step_by(N_STEP_OBS).collect();

    Truth { time, data }
}

fn async_obs(truth: Truth) -> Truth {
    let mut index: Vec<Vec<usize>> = vec![vec![0; N_OBS_STATION]; N_OBS_TIME];
    let mut mask: Vec<Vec<bool>> = vec![vec![true; N_OBS_STATION]; N_TRUTH_TIME];
    let mut rng = rand::thread_rng();

    for i in index.iter_mut().skip(1) {
        for j in i.iter_mut() {
            *j = rng.gen_range(0..10);
        }
    }

    for i in mask[0].iter_mut() {
        *i = false;
    }

    for i in 0..N_OBS_TIME {
        for j in 0..N_OBS_STATION {
            let start = N_STEP_OBS * i + N_STEP_OBS / 2;
            let end = start + N_STEP_OBS;
            let mut ti = rng.gen_range(start..end);
            if ti >= N_TRUTH_TIME {
                ti = N_TRUTH_TIME - rng.gen_range(1..=6);
            }
            mask[ti][j] = false;
        }
    }

    let masked_data: Vec<Vec<f64>> = truth
        .data
        .into_iter()
        .zip(mask.into_iter())
        .map(|(d, m)| {
            d.into_iter()
                .zip(m.into_iter())
                .map(|(d, m)| match m {
                    true => f64::NAN,
                    false => d,
                })
                .collect()
        })
        .collect();

    let truth = Truth {
        data: masked_data,
        ..truth
    };

    shrink_truth(truth)
}

fn missing_obs(truth: Truth) -> Truth {
    const N_MISSING_STATION: usize = 1;
    const PERIOD_MISSING: usize = N_OBS_TIME / 2;
    let mut truth = truth;
    // station, start, period
    let mut miss_index: Vec<(usize, usize, usize)> = vec![(0, 0, 0); N_MISSING_STATION];
    let mut rng = rand::thread_rng();

    for mi in 0..miss_index.len() {
        // low effect, but should be faster than shuffle in low N
        let index = loop {
            let i = rng.gen_range(0..N_OBS_STATION);
            if miss_index
                .iter()
                .take(mi)
                .filter(|(mi, _, _)| *mi == i)
                .count()
                == 0
            {
                break i;
            }
        };

        let period = PERIOD_MISSING;
        let start = rng.gen_range(0..truth.time.len() - period);
        miss_index[mi] = (index, start, period);
    }

    for mi in miss_index.into_iter() {
        let (index, start, period) = mi;
        truth.data[index]
            .iter_mut()
            .skip(start)
            .filter(|d| !d.is_nan())
            .take(period)
            .for_each(|d| *d = f64::NAN);
    }

    shrink_truth(truth)
}

fn error_obs(truth: Truth, sd: f64) -> Truth {
    const N_ERROR_STATION: usize = 1;
    const PERIOD_ERROR: usize = N_OBS_TIME / 2;
    const ERROR_AMP: f64 = 5.0;
    let mut truth = truth;
    // station, start, period, error
    let mut error_index: Vec<(usize, usize, usize)> = vec![(0, 0, 0); N_ERROR_STATION];
    let normal = Normal::new(0.0, sd * ERROR_AMP).unwrap();
    let mut rng = rand::thread_rng();

    for mi in 0..error_index.len() {
        // low effect, but should be faster than shuffle in low N
        let index = loop {
            let i = rng.gen_range(0..N_OBS_STATION);
            if error_index
                .iter()
                .take(mi)
                .filter(|(mi, _, _)| *mi == i)
                .count()
                == 0
            {
                break i;
            }
        };

        let period = PERIOD_ERROR;
        let start = rng.gen_range(0..truth.time.len() - period);
        error_index[mi] = (index, start, period);
    }

    for mi in error_index.into_iter() {
        let (index, start, period) = mi;
        let err = normal.sample_iter(&mut rng);
        truth.data[index]
            .iter_mut()
            .zip(err)
            .skip(start)
            .filter(|(d, _)| !d.is_nan())
            .take(period)
            .for_each(|(d, e)| *d += e);
    }

    shrink_truth(truth)
}

fn shrink_truth(truth: Truth) -> Truth {
    let (shrinked_time, shrinked_data): (Vec<f64>, Vec<Vec<f64>>) = truth
        .time
        .into_iter()
        .zip(truth.data.into_iter())
        .filter(|(_, d)| d.iter().filter(|d| !d.is_nan()).count() != 0)
        .unzip();

    Truth {
        time: shrinked_time,
        data: shrinked_data,
    }
}

fn write_to_txt(file: &str, truth: Truth) -> DataResult<()> {
    let mut f = File::create(file)?;
    let (time, data) = (truth.time, truth.data);
    for (t, d) in time.into_iter().zip(data.into_iter()) {
        let line = d
            .into_iter()
            .fold(t.to_string(), |t, d| t + " " + &d.to_string());
        writeln!(&mut f, "{}", line)?;
    }

    Ok(())
}

fn write_truth(file: &str, truth: Truth) -> DataResult<()> {
    let mut f = File::create(file)?;
    let (time, data) = (truth.time, truth.data);
    for (_, d) in time.into_iter().zip(data.into_iter()) {
        let line = d
            .into_iter()
            .fold("".to_string(), |t, d| t + " " + &d.to_string());
        writeln!(&mut f, "{}", line)?;
    }

    Ok(())
}
