#![feature(convert)]
#![feature(core)]
#![feature(split_off)]
#![feature(iter_arith)]
#![feature(drain)]
#![feature(vec_push_all)]

extern crate getopts;
extern crate core;

extern crate piston;
extern crate graphics;
extern crate glutin_window;
extern crate opengl_graphics;

mod display;


use getopts::Options;
use std::env;
use std::thread::{JoinHandle, Thread};
use std::thread;
use std::sync::{Arc, Mutex};
use std::collections::{HashMap, BinaryHeap};
use std::sync::mpsc::{channel, Receiver, Sender};

extern crate bounded_spsc_queue;
use bounded_spsc_queue::{Producer, Consumer};

extern crate kissfft;
use kissfft::KissFFT;
use kissfft::binding::kiss_fft_cpx;

extern crate sound_stream;
//use sound_stream::{CallbackFlags, CallbackResult, SoundStream, Settings, StreamParams};
use sound_stream::*;
extern crate osc;
use osc::osc_sender::*;
use osc::osc_data::*;
use osc::osc_data::OscPacket::*;
use osc::osc_data::OscArg::*;
use std::net::{Ipv4Addr,SocketAddrV4};
use core::str::FromStr;

type ChannelBuffers = Arc<Mutex<HashMap<i32, Vec<f32>>>>;

const Fs : usize = 44100usize;
const osc_interval : usize = Fs / 30usize;

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
fn unindex_magnitudes(mags : &Vec<IndexedMagnitude>) -> Vec<f32> {
    let mut unindexed = vec!();
    for mag in mags {
        unindexed.push(mag.magnitude);
    }
    unindexed
}
extern crate num;
use num::integer::Integer;
fn pitch_detect(buckets : &Vec<f32>) -> Option<f32> {
    //  1 indexed
    //  filter to peaks
    if let Some(peak) = first_peak(&index_magnitudes(buckets), 1./10.) {
        return Some((peak.index * Fs) as f32 / (buckets.len() as f32 * 2.0));
    }
    None
}

fn idx_to_freq(idx : usize, N : usize) -> f32 {
    (idx * Fs) as f32 / N as f32
}

fn bkt_to_freq(bkt : &IndexedMagnitude, buckets : usize) -> f32 {
    idx_to_freq(bkt.index, buckets)
}

fn dissonance(buckets : &Vec<f32>) -> f32{
    let (peaks, _) = find_peaks(&index_magnitudes(buckets), 0.1);
    let (second_order_peaks, _) = find_peaks(&peaks, 0.1);
    return second_order_peaks.len() as f32;
}

fn norm(v : &Vec<f32>) -> Vec<f32> {
    let mut mag = 0f32;
    for e in v {
        mag += *e * *e;
    }
    mag = mag.sqrt();
    let mut norm_v = vec!();
    for e in v {
        norm_v.push(*e / mag);
    }
    norm_v
}

fn normalized_dot(v1 : &Vec<f32>, v2 : &Vec<f32>) -> f32 {
    if v1.len() != v2.len() {
        return 0f32;
    }
    let mut dot = 0f32;
    let v1 = norm(v1);
    let v2 = norm(v2);
    for (e1, e2) in v1.iter().zip(v2.iter()) {
        dot += e1 * e2;
    }
    dot
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
    let mut buckets = buckets.iter();
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

const PEAK_THRESHOLD : f32 = 0.01;
fn zero_peak(buckets : &mut Vec<IndexedMagnitude>, idx : usize) {
    //  backwards
    if idx > 0 {
        let mut low = buckets[idx].clone();
        let mut low_idx = idx - 1;
        while low_idx >= 0 {
            if buckets[low_idx].magnitude < low.magnitude {
                low = buckets[low_idx].clone();
            }
            if buckets[low_idx].magnitude - low.magnitude > PEAK_THRESHOLD {
                //  end of peak
                buckets[low_idx].magnitude = 0.;
                break;
            }
            buckets[low_idx].magnitude = 0.;
            if low_idx == 0 {
                break;
            }
            low_idx -= 1;
        }
    }

    //  forwards
    if idx < buckets.len() - 1 {
        let mut low = buckets[idx].clone();
        let mut low_idx = idx + 1;
        while low_idx < buckets.len() {
            if buckets[low_idx].magnitude < low.magnitude {
                low = buckets[low_idx].clone();
            }
            if buckets[low_idx].magnitude - low.magnitude > PEAK_THRESHOLD {
                //  end of peak
                buckets[low_idx].magnitude = 0.;
                break;
            }
            buckets[low_idx].magnitude = 0.;
            if low_idx == buckets.len() - 1 {
                break;
            }
            low_idx += 1;
        }
    }

    buckets[idx].magnitude = 0.;
}

//  zeros all of the peaks located at each integer multiple of 'idx'
fn zero_harmonic(buckets : &mut Vec<IndexedMagnitude>, peaks : &Vec<IndexedMagnitude>, valleys : &Vec<Option<IndexedMagnitude>>, idx : usize) {
    let mut h_idx = idx + 1;
    //println!("{:?}\n{:?}", peaks, valleys);
    for i in 1..(buckets.len() / h_idx) {
        //  for each harmonic of the idx, erase between the two adjacent valleys
        //  first, binary search for peak
        let mut low_valley_idx = 0usize;
        match valleys.binary_search_by(
            |i_m| match(i_m) {
                &Some(ref i_m) => {
                    return i_m.index.cmp(&(h_idx * i));
                },
                &None => {
                    //  valley at idx 0
                    return Ordering::Less;
                }
            }) {
            Ok(idx) => {
                let mut idx = idx;
                if idx > 0 {
                    idx -= 1;
                }
                low_valley_idx = idx;
            },
            Err(idx) => {
                let mut idx = idx;
                if idx > 0 {
                    idx -= 1;
                }
                low_valley_idx = idx;
            }
        };
        if low_valley_idx >= valleys.len() - 1 {
            low_valley_idx = valleys.len() - 1;
        }
        let low_idx = match(&valleys[low_valley_idx]) {
            &Some(ref low_valley) => low_valley.index.clone(),
            &None => 0usize,
        };
        let mut valleys = valleys.iter().skip(low_valley_idx);
        let low_idx = match(valleys.next()) {
            Some(valley) => {
                if let &Some(ref valley) = valley {
                    //  valley exists
                    valley.index
                } else {
                    //  first valley
                    0usize
                }
            },
            None =>
                //  out of bounds
                return
        };
        let high_idx = match(valleys.next()) {
            Some(&Some(ref valley)) => {
                valley.index
            },
            _ => low_idx
        };
        for i in low_idx..high_idx+1 {
            buckets[i].magnitude = 0.;
        }
        //println!("{} {} {}", low_idx, h_idx*i, high_idx);
    }
}

#[macro_use]
extern crate itertools;

use itertools::Itertools;

const LOW_PEAK_ELIMINATION_IDX_THRESHOLD : usize = 10usize;
fn remove_harmonics(buckets : &Vec<IndexedMagnitude>, num : usize) -> Vec<IndexedMagnitude> {
    let mut buckets = buckets.clone();
    for i in 0..num {
        //  remove harmonics of biggest mountain on each iteration
        let (peaks, valleys) = find_peaks(&buckets, 1./1000.);
        let (mtns, _) = find_peaks(&peaks, 1./50.);
        if let Some(mtn) = mtns.iter().cloned()
            .fold1(|l, r|
                   if l.magnitude > r.magnitude {
                       l
                   } else {
                       r
                   }) {
                       zero_harmonic(&mut buckets, &mtns, &valleys, mtn.index);
        } else {
            break;
        }
    }
    buckets
}

fn pitch_centroid(buckets : &Vec<f32>) -> f32 {
    let mut centroid = 0f32;
    let totalMag : f32 = buckets.iter().sum();
    let mut i = 1usize;
    for bucket in buckets {
        centroid += idx_to_freq(i, buckets.len() * 2) *  bucket / totalMag;
        i += 1usize;
    }
    centroid
}

fn compute_harmonicity(buckets : &Vec<f32>) -> f32 {
    let total_energy : f32 = buckets.iter().sum();
    let indexed_buckets = index_magnitudes(buckets);
    let filtered_buckets = remove_harmonics(&indexed_buckets, 7);
    let inharmonic_energy : f32 = unindex_magnitudes(&filtered_buckets).iter().sum();
    return 1. - inharmonic_energy / total_energy;
}

fn zero_padded_fft_norm(buf : &Vec<kiss_fft_cpx>, zeros : usize) -> Vec<f32>{
    let mut zero_vec = vec!();
    for i in 0..zeros {
        zero_vec.push(kiss_fft_cpx{r : 0f32, i : 0f32});
    }
    let mut fft_buf = vec!();
    fft_buf.push_all(&zero_vec[..]);
    fft_buf.push_all(&buf[..]);
    let mut fft = KissFFT::new(buf.len() + zeros, false);
    let mut fft_out = fft.transform_norm_to_vec(&fft_buf[..]);
    let N = fft_out.len();
    fft_out.split_off(N / 2usize);
    Vec::from_iter(fft_out.iter().map(|c| (c.r * c.r + c.i * c.i).sqrt()))
}

use std::iter::FromIterator;
fn meter_fft(
    num_channels : i32,
    c : Consumer<f32>,
    osc_sender : Arc<Mutex<OscSender>>,
    osc_prefix : String,
    display_buckets : Arc<Mutex<Vec<f32>>>,
    display_lines : Arc<Mutex<Vec<([f32; 4], usize)>>>,
    ) -> JoinHandle<()> {
        let mut bufs_by_channel = HashMap::new();
        let mut chan_index = 0;
        let fft_buckets = 8192/4;
        let zero_pad_coef = 8192 / fft_buckets;
        let N = fft_buckets * zero_pad_coef;
        let mut samples = 0usize;
        let mut last_sent_time = 0usize;
        thread::spawn(move || {
            loop {
                if let Some(s) = c.try_pop() {
                    let mut buf = bufs_by_channel.entry(chan_index).or_insert(vec!());
                    buf.push(kiss_fft_cpx{r : s, i : 0f32});
                    if buf.len() == fft_buckets {
                        let fft_norm = zero_padded_fft_norm(buf, N - fft_buckets);
                        let peaks = find_peaks(&index_magnitudes(&fft_norm), 1./12.).0.len();
                        let detected_pitch = pitch_detect(&fft_norm);
                        let harmonicity = compute_harmonicity(&fft_norm);
                        if chan_index == 0 {
                            let mut display_buckets = display_buckets.lock().unwrap();
                            display_buckets.clear();
                            let removed_first = remove_harmonics(&mut index_magnitudes(&fft_norm), 7);
                            display_buckets.push_all(&unindex_magnitudes(&removed_first)[..]);
                            let YELLOW = [243./255., 232./255., 51./255., 0.5];
                            let RED = [239./255., 101./255., 68./255., 0.5];
                            let (peaks, valleys) = find_peaks(&index_magnitudes(&fft_norm), 1./1000.);
                            let (mtns, _) = find_peaks(&peaks, 1./50.);
                            let mut display_lines = display_lines.lock().unwrap();
                            display_lines.clear();
                            for mtn in mtns.iter() {
                                display_lines.push((RED.clone(), mtn.index.clone()));
                            }
                            for valley in valleys.iter() {
                                if let &Some(ref valley) = valley {
                                    display_lines.push((YELLOW.clone(), valley.index.clone()));
                                }
                            }
                            //display_buckets.push_all(&fft_norm[..]);
                            //display_buckets.push_all(&ordered_harmonics[..]);
                        }
                        if samples > last_sent_time + osc_interval * (num_channels as usize) {
                            last_sent_time = samples;
                            let mut sender = osc_sender.lock().unwrap();
                            let mut msgs = vec!();
                            if let Some(detected_pitch) = detected_pitch {
                                msgs.push(
                                    OscMessage{
                                        addr : format!("/opera/meter/{}/track{}/detectedPitch", osc_prefix, chan_index).to_string(),
                                        args : vec!(OscFloat(detected_pitch))
                                    }
                                    );
                            }
                            msgs.push(
                                OscMessage{
                                    addr : format!("/opera/meter/{}/track{}/harmonicity", osc_prefix, chan_index).to_string(),
                                    args : vec!(OscFloat(harmonicity))
                                },
                                );
                            msgs.push(
                                OscMessage{
                                    addr : format!("/opera/meter/{}/track{}/numPeaks", osc_prefix, chan_index).to_string(),
                                    args : vec!(OscInt(peaks as i32))
                                }
                                );
                            sender.send(
                                OscBundle{
                                    time_tag : (0, 1),
                                    conts: msgs
                                });
                        }
                        buf.clear();
                    }

                    chan_index = (chan_index + 1) % num_channels;
                    samples += 1usize;
                } else {
                    thread::sleep_ms(12u32);
                }
            }
        })

    }
fn meter_rms(
    num_channels : i32,
    c : Consumer<f32>,
    p : Producer<f32>,
    osc_sender : Arc<Mutex<OscSender>>,
    osc_prefix : String
    ) -> JoinHandle<()> {
    let mut rms_map : HashMap<i32, f32> = HashMap::new();
    let mut max_amp_map : HashMap<i32, f32> = HashMap::new();
    let mut chan_idx = 0;
    let mut samples = 0usize;
    let mut last_sent_time = 0usize;
    // TODO: use sample rate
    let alpha = 30f32 / 44100f32;
    thread::spawn(move || {
        loop {
            if let Some(s) = c.try_pop() {
                let old_rms = *rms_map.entry(chan_idx).or_insert(0f32);
                let new_rms = ((old_rms.powi(2) + s.powi(2)) / 2f32).sqrt();
                rms_map.insert(chan_idx, (1f32 - alpha) * old_rms + new_rms * alpha);

                let old_max_amp = *max_amp_map.entry(chan_idx).or_insert(0f32);
                max_amp_map.insert(chan_idx, f32::max(old_max_amp, f32::abs(s)));
                if samples > last_sent_time + osc_interval * (num_channels as usize) {
                    let mut sender = osc_sender.lock().unwrap();
                    let mut msg_vec = vec!();
                    for (chan, rms) in rms_map.iter() {
                        let rms_msg = OscMessage{
                            addr : format!("/opera/meter/{}/track{}/rms", osc_prefix, chan).to_string(),
                            args : vec!(OscFloat(*rms as f32))
                        };
                        msg_vec.push(rms_msg);
                    }
                    for (chan, max_amp) in max_amp_map.iter() {
                        let amp_msg = OscMessage{
                            addr : format!("/opera/meter/{}/track{}/maxAmp", osc_prefix, chan).to_string(),
                            args : vec!(OscFloat(*max_amp))
                        };
                        msg_vec.push(amp_msg);
                    }
                    max_amp_map.clear();
                    let bundle = OscBundle {
                        time_tag : (0, 1),
                        conts : msg_vec
                    };
                    if let Err(e) = sender.send(bundle) {
                        println!("Error sending OSC: {:?}", e);
                    }
                    last_sent_time = samples;
                }
                chan_idx = (chan_idx + 1) % num_channels;
                samples += 1usize;
                p.push(s);
            } else {
                thread::sleep_ms(1u32);
            }
        }
    })
}

fn setup_stream(
    frames_per_buffer : u16,
    active_channels : i32,
    total_channels : i32,
    p : Producer<f32>) {
        let mut channel_idx = 0usize;
        let f = Box::new(move |input: &[f32], _: Settings, o: &mut[f32], _: Settings, dt: f64, _: CallbackFlags| {
            for s in input.iter() {
                if (channel_idx as i32) < active_channels {
                    p.push(*s);
                }
                channel_idx = (channel_idx + 1usize) % total_channels as usize;
            }
            return CallbackResult::Continue;
        });

        let stream = SoundStream::new().sample_hz(Fs as f64).frames_per_buffer(frames_per_buffer).duplex(StreamParams::new().channels(total_channels), StreamParams::new()).run_callback(f).unwrap();
        while let Ok(true) = stream.is_active() {
            thread::sleep_ms(500);
        }
}

fn main() {
     let args: Vec<String> = env::args().collect();
     let active_channels : i32 = i32::from_str_radix(args[1].as_str(), 10).unwrap();
     let total_channels : i32 = i32::from_str_radix(args[2].as_str(), 10).unwrap();
     let frames_per_buffer : usize = usize::from_str_radix(args[3].as_str(), 10).unwrap();
     let send_ip = args[4].as_str();
     let send_port = u16::from_str_radix(args[5].as_str(), 10).unwrap();
     let meter_id = args[6].as_str();

     let (p_pa, c_rms) = bounded_spsc_queue::make(16 * frames_per_buffer * (total_channels as usize));
     let (p_rms, c_fft) = bounded_spsc_queue::make(16 * frames_per_buffer * (total_channels as usize));

     let mut sender;
     match OscSender::new(
         SocketAddrV4::new(Ipv4Addr::from_str("0.0.0.0").unwrap(), 0),
         SocketAddrV4::new(Ipv4Addr::from_str(send_ip).unwrap(), send_port)
         ) {
         Ok(s) => { sender = s; },
         Err(e) => { panic!(e); }
     }
     let sender_arc = Arc::new(Mutex::new(sender));
     let fft_magnitudes = Arc::new(Mutex::new(vec!()));
     let vertical_lines = Arc::new(Mutex::new(vec!()));
     meter_rms(active_channels, c_rms, p_rms, sender_arc.clone(), String::from(meter_id));
     meter_fft(active_channels, c_fft, sender_arc.clone(), String::from(meter_id), fft_magnitudes.clone(), vertical_lines.clone());
     thread::spawn(move || setup_stream(frames_per_buffer as u16, active_channels, total_channels, p_pa));
     //display::init(fft_magnitudes, vertical_lines);
     loop {
        thread::sleep_ms(500);
     }
}
