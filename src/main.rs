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

#[derive(Clone)]
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
extern crate num;
use num::integer::Integer;
fn pitch_detect(buckets : &Vec<f32>) -> f32{
    //  gcd of top 5 bands
    let mut indexed_buckets = vec!();
    let mut i = 0usize;
    for bucket in buckets {
        indexed_buckets.push(IndexedMagnitude{index : i, magnitude: bucket.clone()});
        i += 1usize;
    }
    let mut harmonics = indexed_buckets.clone();
    for indexed_bucket in indexed_buckets.iter() {
        for i in 1..4 {
            if indexed_bucket.index / i > 0usize {
                harmonics[indexed_bucket.index / i].magnitude += indexed_bucket.magnitude;
            }
        }
    }
    harmonics.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let max_idx = harmonics.pop().unwrap();
    //  convert index to freq
    let freq = (max_idx.index * Fs) as f32 / buckets.len() as f32;
    return freq;
}

fn idx_to_freq(idx : usize, buckets : usize) -> f32 {
    (idx * Fs) as f32 / buckets as f32
}

fn bkt_to_freq(bkt : &IndexedMagnitude, buckets : usize) -> f32 {
    idx_to_freq(bkt.index, buckets)
}

fn dissonance(buckets : &Vec<f32>) -> f32{
    let mut indexed_buckets = vec!();
    let mut i = 0usize;
    for bucket in buckets {
        indexed_buckets.push(IndexedMagnitude{index : i, magnitude: bucket.clone()});
        i += 1usize;
    }
    //  desc sort by magnitude
    indexed_buckets.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    let f0 = bkt_to_freq(&indexed_buckets[0], buckets.len() * 2);
    let f1 = bkt_to_freq(&indexed_buckets[1], buckets.len() * 2);
    let f2 = bkt_to_freq(&indexed_buckets[2], buckets.len() * 2);
    let mut d1 = f1 / f0;
    let mut d2 = f2 / f0;
    //println!("{} {} {}", f0, f1, f2);
    d1 = (d1 - d1.floor() - 0.5f32) * 2f32;
    d2 = (d2 - d2.floor() - 0.5f32) * 2f32;
    (d1 - d2).abs()
}

/// Return peaks of the signal considering values above 1/5 of the maximum
/// Expects a vector of magnitudes
fn find_peaks(buckets : &Vec<f32>) -> Vec<f32> {
    let mut threshold = 0.0;
    for bucket in buckets.iter() {
        threshold = f32::max(threshold, *bucket);
    }
    threshold /= 5.0;
    let mut peaks = vec!();
    let mut buckets = buckets.iter();
    let mut left = buckets.next().unwrap();
    let mut mid = buckets.next().unwrap();
    for bucket in buckets {
        if *bucket < threshold {
            continue;
        }
        if left < mid && bucket < mid {
            peaks.push(*bucket);
        }
        left = mid;
        mid = bucket;
    }
    peaks
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

use std::iter::FromIterator;
fn meter_fft(
    num_channels : i32,
    c : Consumer<f32>,
    osc_sender : Arc<Mutex<OscSender>>,
    osc_prefix : String,
    display_buckets : Arc<Mutex<Vec<f32>>>,
    ) -> JoinHandle<()> {
        let mut bufs_by_channel = HashMap::new();
        let mut chan_index = 0;
        let fft_buckets = 1024;
        let mut samples = 0usize;
        let mut last_sent_time = 0usize;
        thread::spawn(move || {
            loop {
                if let Some(s) = c.try_pop() {
                    let mut buf = bufs_by_channel.entry(chan_index).or_insert(vec!());
                    buf.push(kiss_fft_cpx{r : s, i : 0f32});
                    if buf.len() == fft_buckets {
                        let mut fft = KissFFT::new(fft_buckets, false);
                        let mut fft_out = fft.transform_to_vec(buf);
                        //  only use first half
                        fft_out.split_off(fft_buckets / 2usize);
                        let fft_norm = Vec::from_iter(fft_out.iter().map(|c| (c.r * c.r + c.i * c.i).sqrt() * 1f32 / (fft_buckets as f32).sqrt()));
                        let peaks = find_peaks(&fft_norm).len();
                        if chan_index == 0 {
                            let mut display_buckets = display_buckets.lock().unwrap();
                            display_buckets.clear();
                            display_buckets.push_all(&fft_norm[..]);
                        }
                        let dissonance = dissonance(&fft_norm);
                        let centroid = pitch_centroid(&fft_norm);
                        if samples > last_sent_time + osc_interval * (num_channels as usize) {
                            last_sent_time = samples;
                            let mut sender = osc_sender.lock().unwrap();
                            sender.send(
                                OscBundle{
                                    time_tag : (0, 1),
                                    conts: vec!(
                                        OscMessage{
                                            addr : format!("/opera/meter/{}/track{}/pitchCentroid", osc_prefix, chan_index).to_string(),
                                            args : vec!(OscFloat(centroid))
                                        },
                                        OscMessage{
                                            addr : format!("/opera/meter/{}/track{}/dissonance", osc_prefix, chan_index).to_string(),
                                            args : vec!(OscFloat(dissonance))
                                        },
                                        OscMessage{
                                            addr : format!("/opera/meter/{}/track{}/numPeaks", osc_prefix, chan_index).to_string(),
                                            args : vec!(OscInt(peaks as i32))
                                        },
                                        )
                                });
                        }
                        //  sliding 1/8 window
                        buf.drain(..fft_buckets * 7/8);
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
     meter_rms(active_channels, c_rms, p_rms, sender_arc.clone(), String::from(meter_id));
     meter_fft(active_channels, c_fft, sender_arc.clone(), String::from(meter_id), fft_magnitudes.clone());
     thread::spawn(move || setup_stream(frames_per_buffer as u16, active_channels, total_channels, p_pa));
     //display::init(fft_magnitudes);
     loop {
        thread::sleep_ms(500);
     }
}
