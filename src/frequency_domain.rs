use bounded_spsc_queue::{Consumer};
use kissfft::KissFFT;
use kissfft::binding::kiss_fft_cpx;
use osc::osc_sender::*;
use osc::osc_data::OscPacket::*;
use osc::osc_data::OscArg::*;
use std::thread;
use std::thread::{JoinHandle};
use std::sync::{Arc, Mutex};
use itertools::Itertools;

const OSC_PER_SEC : usize = 30usize;

#[derive(Clone, Debug)]
struct IndexedMagnitude {
    index : usize,
    magnitude : f32
}

use std::cmp::Ordering;
impl PartialOrd for IndexedMagnitude {
     fn partial_cmp(&self, other: &IndexedMagnitude) -> Option<Ordering> {
        return self.magnitude.partial_cmp(&other.magnitude);
     }
}
impl PartialEq for IndexedMagnitude {
    fn eq(&self, other: &IndexedMagnitude) -> bool {
        return self.magnitude.eq(&other.magnitude);
    }
}

fn index_magnitudes(mags : &Vec<f32>) -> Vec<IndexedMagnitude> {
    let mut indexed = vec!();
    let mut i = 0usize;
    for mag in mags {
        indexed.push(IndexedMagnitude{index : i, magnitude : *mag});
        i += 1usize;
    }
    indexed
}
use std::f32;
const PEAK_THRESHOLD : f32 = 9.5/10.;
fn pitch_detect_ac(buckets : &Vec<f32>, fs : usize) -> Option<(f32, usize)> {
    //  1 indexed
    //  filter to peaks
    let buckets = &mut buckets.clone()[0..buckets.len()/4].to_vec();
    for i in 0..buckets.len() {
        buckets[i] = f32::max(0., buckets[i]);
    }
    let mut first_zero_idx = 0usize;
    for i in 0..buckets.len() {
        if buckets[i] <= 0. {
            first_zero_idx = i;
            break;
        }
    }
    let peaks = find_peaks(&index_magnitudes(buckets)[first_zero_idx..buckets.len()].to_vec(), PEAK_THRESHOLD).0;
    for peak in peaks.iter() {
        if peak.index > first_zero_idx {
            return Some((fs as f32 / 2. / peak.index as f32, peak.index));
        }
    }
    None
}

fn idx_to_freq(idx : usize, n : usize, fs : usize) -> f32 {
    (idx * fs) as f32 / n as f32
}

fn bkt_to_freq(bkt : &IndexedMagnitude, buckets : usize, fs : usize) -> f32 {
    idx_to_freq(bkt.index, buckets, fs)
}

fn compute_dissonance(buckets : &Vec<f32>, fs : usize) -> f32{
    let (mut peaks, _) = find_peaks(&find_peaks(&index_magnitudes(buckets), 1./12.).0, 1./12.);
    peaks.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    if peaks.len() < 3 {
        return 0.;
    }
    let f1 = bkt_to_freq(&peaks[0], buckets.len() * 2, fs);
    let f2 = bkt_to_freq(&peaks[1], buckets.len() * 2, fs);
    let f3 = bkt_to_freq(&peaks[2], buckets.len() * 2, fs);
    let d1 = f2 / f1;
    let d2 = f3 / f1;

    let fd1 = (d1 - d1.floor() - 0.5).abs() * 2.;
    let fd2 = (d2 - d2.floor() - 0.5).abs() * 2.;
    return (fd1 - fd2).abs();
}

/// Return peaks of the signal considering values above threshold * the maximum
/// Expects a vector of magnitudes
/// Also returns the valleys
fn find_peaks(buckets : &Vec<IndexedMagnitude>,
              threshold_coef : f32) -> (Vec<IndexedMagnitude>,
                                        Vec<Option<IndexedMagnitude>>) {
    let mut threshold = 0.0;
    for bucket in buckets.iter() {
        threshold = f32::max(threshold, bucket.magnitude);
    }
    threshold *= threshold_coef;
    let mut peaks = vec!();
    let mut valleys = vec!();
    let buckets = buckets.iter();
    let mut high : Option<IndexedMagnitude> = None;
    let mut low : Option<IndexedMagnitude> = None;
    let mut pre_peak_low : Option<IndexedMagnitude> = None;
    for bucket in buckets {
        if let Some(low_val) = low.clone() {
           if bucket.magnitude < low_val.magnitude {
               low = Some(bucket.clone());
           }
        } else {
            low = Some(bucket.clone());
        }
        if let Some(high_val) = high.clone() {
           if bucket.magnitude > high_val.magnitude {
               high = Some(bucket.clone());
               pre_peak_low = low.clone();
           }
        } else {
            high = Some(bucket.clone());
            pre_peak_low = low.clone();
        }
        if let Some(high_val) = high.clone() {
            if high_val.magnitude - bucket.magnitude > threshold / 2.
                && bucket.magnitude - low.clone().unwrap().magnitude > threshold / 2. {
                    peaks.push(high_val.clone());
                    valleys.push(pre_peak_low.clone());
                    high = None;
                    low = None;
                    pre_peak_low = None;
            }
        }
    }
    (peaks, valleys)
}

fn zero_padded_fft_norm(buf : &Vec<kiss_fft_cpx>, zeros : usize) -> Vec<f32>{
    let mut zero_vec = vec!();
    for _ in 0..zeros {
        zero_vec.push(kiss_fft_cpx{r : 0f32, i : 0f32});
    }
    let mut fft_buf = vec!();
    fft_buf.push_all(&zero_vec[..]);
    fft_buf.push_all(&buf[..]);
    let mut fft = KissFFT::new(buf.len() + zeros, false);
    let mut fft_out = fft.transform_norm_to_vec(&fft_buf[..]);
    let n = fft_out.len();
    fft_out.split_off(n / 2usize);
    Vec::from_iter(fft_out.iter().map(|c| (c.r * c.r + c.i * c.i).sqrt()))
}

fn fft_inv_real(buf : &Vec<kiss_fft_cpx>) -> Vec<f32>{
    let mut fft = KissFFT::new(buf.len(), true);
    let fft_out = fft.transform_norm_to_vec(&buf[..]);
    fft_out.into_iter().map(|r| r.r).collect()
}

use std::f32::consts;
fn hann_window(v : Vec<kiss_fft_cpx>) -> Vec<kiss_fft_cpx> {
    let mut v = v;
    for i in 0..v.len() {
        v[i] = kiss_fft_cpx{
            r : v[i].r * ((consts::PI * i as f32) / (v.len() as f32 - 1.)).sin().powi(2),
            i : v[i].i * ((consts::PI * i as f32) / (v.len() as f32 - 1.)).sin().powi(2)
        };
    }
    v
}

fn autocorrelate(buf : &Vec<kiss_fft_cpx>) -> Vec<f32> {
    let buf = hann_window(buf.clone());
    let mut fft_norm : Vec<f32> = zero_padded_fft_norm(&buf, buf.len()).into_iter().map(|r| r*r).collect();
    fft_norm[0] = 0f32;
    let ac = fft_inv_real(&fft_norm.into_iter().map(|r| kiss_fft_cpx{r:r, i:0f32}).collect());
    let first = ac[0];
    let ac = ac.into_iter().map(|e| e/first).collect();
    ac
}

use std::iter::FromIterator;
pub fn meter_fft(
    sampling_frequency : usize,
    active_channels : Vec<String>,
    c : Consumer<f32>,
    osc_sender : Arc<Mutex<OscSender>>,
    osc_prefix : String,
    display_buckets : Arc<Mutex<Vec<f32>>>,
    display_lines : Arc<Mutex<Vec<([f32; 4], usize)>>>,
    display_chan_index : Option<i32>,
    ) -> JoinHandle<()> {
        let mut chan_index = 0;
        let fft_buckets = 2048;
        let mut samples = 0usize;
        let mut last_sent_time = 0usize;
        let mut bufs_by_channel = Vec::new();
        let mut pitch_by_chan : Vec<f32> = Vec::new();
        let mut last_pitch_by_chan : Vec<(f32, usize)> = Vec::new();
        let mut dissonance_by_chan : Vec<f32> = Vec::new();
        let mut num_peaks_by_chan : Vec<usize> = Vec::new();
        for _ in 0..active_channels.len() {
            bufs_by_channel.push(vec!());
            pitch_by_chan.push(0.);
            last_pitch_by_chan.push((0., 0));
            dissonance_by_chan.push(0.);
            num_peaks_by_chan.push(0);
        }
        thread::spawn(move || {
            loop {
                if let Some(s) = c.try_pop() {
                    let buf = &mut bufs_by_channel[chan_index];
                    buf.push(kiss_fft_cpx{r : s, i : 0f32});
                    if buf.len() == fft_buckets {
                        let fft_norm = autocorrelate(buf);
                        //let fft_norm = zero_padded_cepstrum(&hann_window(buf.clone()), n - fft_buckets);

                        let (peaks, valleys) = find_peaks(&index_magnitudes(&fft_norm), PEAK_THRESHOLD);
                        let (mtns, _) = find_peaks(&peaks, 1./5.);
                        num_peaks_by_chan[chan_index] = mtns.len();
                        if let Some((detected_pitch, index)) = pitch_detect_ac(&fft_norm, sampling_frequency) {
                            let black = [0./255., 0./255., 0./255., 0.5];
                            pitch_by_chan[chan_index] = detected_pitch;
                            let mut display_lines = display_lines.lock().unwrap();
                            display_lines.push((black.clone(), index));
                        }
                        let dissonance = compute_dissonance(&fft_norm, sampling_frequency);
                        dissonance_by_chan[chan_index] = dissonance;
                        if let Some(display_chan_index) = display_chan_index {
                            if display_chan_index == chan_index as i32 {
                                let mut display_buckets = display_buckets.lock().unwrap();
                                display_buckets.clear();
                                display_buckets.push_all(&fft_norm[..]);
                                let yellow = [243./255., 232./255., 51./255., 0.5];
                                let red = [239./255., 101./255., 68./255., 1.];
                                let mut display_lines = display_lines.lock().unwrap();
                                display_lines.clear();
                                for mtn in peaks.iter() {
                                    display_lines.push((red.clone(), mtn.index.clone()));
                                }
                                for valley in valleys.iter() {
                                    if let &Some(ref valley) = valley {
                                        display_lines.push((yellow.clone(), valley.index.clone()));
                                    }
                                }
                                display_buckets.push_all(&fft_norm[..]);
                            }
                        }
                        if samples > last_sent_time + sampling_frequency * active_channels.len() / OSC_PER_SEC {
                            last_sent_time = samples;
                            let mut sender = osc_sender.lock().unwrap();
                            for i in 0..active_channels.len() {
                                let track_title = &active_channels[i];
                                let detected_pitch = pitch_by_chan[i];
                                let pitch_msg =
                                    OscMessage{
                                        addr : format!("/{}/{}/detectedPitch", track_title, osc_prefix).to_string(),
                                        args : vec!(OscFloat(detected_pitch))
                                    };
                                if let Err(e) = sender.send(pitch_msg) {
                                    println!("Error sending OSC: {:?}", e);
                                }
                                let num_peaks = num_peaks_by_chan[i];
                                let peaks_msg =
                                    OscMessage{
                                        addr : format!("/{}/{}/numPeaks", track_title, osc_prefix).to_string(),
                                        args : vec!(OscInt(num_peaks as i32))
                                    };
                                if let Err(e) = sender.send(peaks_msg) {
                                    println!("Error sending OSC: {:?}", e);
                                }
                                let dissonance = dissonance_by_chan[i];
                                let diss_msg =
                                    OscMessage{
                                        addr : format!("/{}/{}/dissonance", track_title, osc_prefix).to_string(),
                                        args : vec!(OscFloat(dissonance))
                                    };
                                if let Err(e) = sender.send(diss_msg) {
                                    println!("Error sending OSC: {:?}", e);
                                }

                            }
                        }
                        let drain = buf.len() * 3 / 4;
                        for _ in 0..drain {
                            buf.remove(0);
                        }
                        //buf.clear();
                    }
                    samples += 1usize;
                    chan_index = (chan_index + 1) % active_channels.len();
                } else {
                    thread::sleep_ms(12u32);
                }
            }
        })

    }
