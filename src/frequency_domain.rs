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

fn pitch_detect(buckets : &Vec<f32>, fs : usize) -> Option<f32> {
    //  1 indexed
    //  filter to peaks
    if let Some(peak) = first_peak(&find_peaks(&index_magnitudes(buckets), 1./10.).0, 1./10.) {
        return Some((peak.index * fs) as f32 / (buckets.len() as f32 * 2.0));
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

fn first_peak(buckets : &Vec<IndexedMagnitude>, threshold_coef : f32) -> Option<IndexedMagnitude> {
    let (peaks, _) = find_peaks(buckets, threshold_coef);
    if peaks.len() >= 1usize {
        return Some(peaks[0].clone());
    }
    None
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
        let fft_buckets = 512;
        let zero_pad_coef = 8192 / 4 / fft_buckets;
        let n = fft_buckets * zero_pad_coef;
        let mut samples = 0usize;
        let mut last_sent_time = 0usize;
        let mut bufs_by_channel = Vec::new();
        let mut pitch_by_chan : Vec<f32> = Vec::new();
        let mut last_pitch_by_chan : Vec<f32> = Vec::new();
        let mut dissonance_by_chan : Vec<f32> = Vec::new();
        let mut num_peaks_by_chan : Vec<usize> = Vec::new();
        for _ in 0..active_channels.len() {
            bufs_by_channel.push(vec!());
            pitch_by_chan.push(0.);
            last_pitch_by_chan.push(0.);
            dissonance_by_chan.push(0.);
            num_peaks_by_chan.push(0);
        }
        thread::spawn(move || {
            loop {
                if let Some(s) = c.try_pop() {
                    let buf = &mut bufs_by_channel[chan_index];
                    buf.push(kiss_fft_cpx{r : s, i : 0f32});
                    if buf.len() == fft_buckets {
                        let fft_norm = zero_padded_fft_norm(&buf, n - fft_buckets);
                        let (peaks, valleys) = find_peaks(&index_magnitudes(&fft_norm), 1./10.);
                        let (mtns, _) = find_peaks(&peaks, 1./10.);
                        num_peaks_by_chan[chan_index] = mtns.len();
                        if let Some(detected_pitch) = pitch_detect(&fft_norm, sampling_frequency) {
                            if detected_pitch == last_pitch_by_chan[chan_index] {
                                pitch_by_chan[chan_index] = detected_pitch;
                            }
                            last_pitch_by_chan[chan_index] = detected_pitch;
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
                                for mtn in mtns.iter() {
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
                        buf.clear();
                    }
                    samples += 1usize;
                    chan_index = (chan_index + 1) % active_channels.len();
                } else {
                    thread::sleep_ms(12u32);
                }
            }
        })

    }
